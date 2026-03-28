"""
Le jeu de la vie
################
Le jeu de la vie est un automate cellulaire inventé par Conway se basant normalement sur une grille infinie
de cellules en deux dimensions. Ces cellules peuvent prendre deux états :
    - un état vivant
    - un état mort
A l'initialisation, certaines cellules sont vivantes, d'autres mortes.
Le principe du jeu est alors d'itérer de telle sorte qu'à chaque itération, une cellule va devoir interagir avec
les huit cellules voisines (gauche, droite, bas, haut et les quatre en diagonale). L'interaction se fait selon les
règles suivantes pour calculer l'itération suivante :
    - Une cellule vivante avec moins de deux cellules voisines vivantes meurt ( sous-population )
    - Une cellule vivante avec deux ou trois cellules voisines vivantes reste vivante
    - Une cellule vivante avec plus de trois cellules voisines vivantes meurt ( sur-population )
    - Une cellule morte avec exactement trois cellules voisines vivantes devient vivante ( reproduction )

Pour ce projet, on change légèrement les règles en transformant la grille infinie en un tore contenant un
nombre fini de cellules. Les cellules les plus à gauche ont pour voisines les cellules les plus à droite
et inversement, et de même les cellules les plus en haut ont pour voisines les cellules les plus en bas
et inversement.

On itère ensuite pour étudier la façon dont évolue la population des cellules sur la grille.
"""
import pygame  as pg
import numpy   as np
from mpi4py import MPI

globCom = MPI.COMM_WORLD.Dup() #communicateur par défaut
nbp     = globCom.size #nombre de processus lancés
rank    = globCom.rank 
name    = MPI.Get_processor_name()


class Grille:
    """
    Grille torique décrivant l'automate cellulaire.
    En entrée lors de la création de la grille :
        - rank est un entier qui correspond au rang du processus dans le communicateur sub_comm
        - nbp est un entier qui sert à découper la grille en blocs de lignes
        - dim est un tuple contenant le nombre de cellules dans les deux directions (nombre lignes, nombre colonnes)
        - init_pattern est une liste de cellules initialement vivantes sur cette grille (les autres sont considérées comme mortes)
        - color_life est la couleur dans laquelle on affiche une cellule vivante
        - color_dead est la couleur dans laquelle on affiche une cellule morte
    Si aucun pattern n'est donné, on tire au hasard quelles sont les cellules vivantes et les cellules mortes
    Exemple :
       grid = Grille( (10,10), init_pattern=[(2,2),(0,2),(4,2),(2,0),(2,4)], color_life=pg.Color("red"), color_dead=pg.Color("black"))
    """
    def __init__(self, rank, nbp, dim,  init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        import random
        ny = dim[0]
        self.ny_glob = ny
        ny_loc = ny//nbp + (1 if rank < ny%nbp else 0)  #diviser la grille en blocs de lignes 
        self.y_start = ny_loc * rank + (ny%nbp if rank >= ny%nbp else 0)  #1 bloc de lignes par processeur
        self.dimensions = (ny_loc+2, dim[1]) #on compte les voisins aux extrémités
        if init_pattern is not None:
            self.cells = np.zeros(self.dimensions, dtype=np.uint8) #grille du processeur vide
            indices_i = [(v[0]-self.y_start-ny+1)%ny for v in init_pattern] #ensemble des coordonnées des cellules initialement vivantes décalées en ligne
            indices_j = [v[1] for v in init_pattern]
            for i, ind_i in enumerate(indices_i):
            	if ind_i >= 0 and ind_i <= ny_loc+1:
            		self.cells[ind_i,indices_j[i]] = 1
        else:
            self.cells = np.random.randint(2, size=dim, dtype=np.uint8)
        self.col_life = color_life
        self.col_dead = color_dead
        #exit(0)

    def compute_next_iteration(self):
        """
        Calcule la prochaine génération de cellules en suivant les règles du jeu de la vie
        """
        # Remarque 1: on pourrait optimiser en faisant du vectoriel, mais pour plus de clarté, on utilise les boucles
        # Remarque 2: on voit la grille plus comme une matrice qu'une grille géométrique. L'indice (0,0) est donc en bas
        #             à gauche de la grille !
        ny = self.dimensions[0]
        nx = self.dimensions[1]
        next_cells = np.empty(self.dimensions, dtype=np.uint8)
        diff_cells = []
        for i in range(1,ny-1):
            i_above = (i-1)%ny
            i_below = (i+1)%ny
            for j in range(nx):
                j_left = (j-1)%nx
                j_right= (j+1)%nx
                voisins_i = [i_above,i_above,i_above, i     , i      , i_below, i_below, i_below] #indices des lignes des voisins
                voisins_j = [j_left ,j      ,j_right, j_left, j_right, j_left , j      , j_right] #indices des colonnes des voisins
                voisines = np.array(self.cells[voisins_i,voisins_j]) #coordonnées des voisins
                nb_voisines_vivantes = np.sum(voisines) #cellules vivantes : cases remplies par un 1
                if self.cells[i,j] == 1: #si la cellule est vivante
                    if (nb_voisines_vivantes < 2) or (nb_voisines_vivantes > 3): #cas de sous/sur-population, la cellule meurt
                        next_cells[i,j] = 0                                     
                        diff_cells.append((i+self.y_start)*nx+j) #ajout des coordonnées à la liste des changements d'état  
                    else:
                        next_cells[i,j] = 1 #sinon elle reste vivante
                #si la cellule est morte
                elif nb_voisines_vivantes == 3: #cas où la cellule morte est entourée exactement de trois vivantes (reproduction)
                    next_cells[i,j] = 1         #naissance de la cellule
                    diff_cells.append((i+self.y_start)*nx+j)
                else:
                    next_cells[i,j] = 0 #sinon elle reste morte
        self.cells = next_cells #maj de l'état des cellules
        return diff_cells #on retourne les changements d'état

    def modify(self, diff):
        nx = self.dimensions[1] #nb colonnes
        for c in diff:
            nr = c//nx #ligne
            nc = c%nx  #colonne
            self.cells[nr, nc] = (1- self.cells[nr, nc]) #modification de l'état de la cellule



class App:
    """
    Cette classe décrit la fenêtre affichant la grille à l'écran
        - geometry est un tuple de deux entiers donnant le nombre de pixels verticaux et horizontaux (dans cet ordre)
        - grid est la grille décrivant l'automate cellulaire (voir plus haut)
    """
    def __init__(self, geometry, grid):
        self.grid = grid
        # Calcul de la taille d'une cellule par rapport à la taille de la fenêtre et de la grille à afficher 
        self.size_x = geometry[1]//grid.dimensions[1]
        self.size_y = geometry[0]//grid.dimensions[0]
        if self.size_x > 4 and self.size_y > 4 :
            self.draw_color=pg.Color('lightgrey')
        else:
            self.draw_color=None
        # Ajustement de la taille de la fenêtre pour bien fitter la dimension de la grille
        self.width = grid.dimensions[1] * self.size_x
        self.height= grid.dimensions[0] * self.size_y
        # Création de la fenêtre à l'aide de tkinter
        self.screen = pg.display.set_mode((self.width,self.height))
        self.canvas_cells = []

    def compute_rectangle(self, i: int, j: int):
        """
        Calcul la géométrie du rectangle correspondant à la cellule (i,j)
        """
        return (self.size_x*j, self.height - self.size_y*(i + 1), self.size_x, self.size_y)

    def compute_color(self, i: int, j: int):
        if self.grid.cells[i,j] == 0:
            return self.grid.col_dead
        else:
            return self.grid.col_life

    def draw(self):
        [self.screen.fill(self.compute_color(i,j),self.compute_rectangle(i,j)) for i in range(self.grid.dimensions[0]) for j in range(self.grid.dimensions[1])]
        if (self.draw_color is not None):
            [pg.draw.line(self.screen, self.draw_color, (0,i*self.size_y), (self.width,i*self.size_y)) for i in range(self.grid.dimensions[0])]
            [pg.draw.line(self.screen, self.draw_color, (j*self.size_x,0), (j*self.size_x,self.height)) for j in range(self.grid.dimensions[1])]
        pg.display.update()


if __name__ == '__main__':
    import time
    import sys

    pg.init()
    dico_patterns = { # Dimension et pattern dans un tuple
        'blinker' : ((5,5),[(2,1),(2,2),(2,3)]),
        'toad'    : ((6,6),[(2,2),(2,3),(2,4),(3,3),(3,4),(3,5)]),
        "acorn"   : ((100,100), [(51,52),(52,54),(53,51),(53,52),(53,55),(53,56),(53,57)]),
        "beacon"  : ((6,6), [(1,3),(1,4),(2,3),(2,4),(3,1),(3,2),(4,1),(4,2)]),
        "boat" : ((5,5),[(1,1),(1,2),(2,1),(2,3),(3,2)]),
        "glider": ((100,90),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
        "glider_gun": ((400,400),[(51,76),(52,74),(52,76),(53,64),(53,65),(53,72),(53,73),(53,86),(53,87),(54,63),(54,67),(54,72),(54,73),(54,86),(54,87),(55,52),(55,53),(55,62),(55,68),(55,72),(55,73),(56,52),(56,53),(56,62),(56,66),(56,68),(56,69),(56,74),(56,76),(57,62),(57,68),(57,76),(58,63),(58,67),(59,64),(59,65)]),
        "space_ship": ((25,25),[(11,13),(11,14),(12,11),(12,12),(12,14),(12,15),(13,11),(13,12),(13,13),(13,14),(14,12),(14,13)]),
        "die_hard" : ((100,100), [(51,57),(52,51),(52,52),(53,52),(53,56),(53,57),(53,58)]),
        "pulsar": ((17,17),[(2,4),(2,5),(2,6),(7,4),(7,5),(7,6),(9,4),(9,5),(9,6),(14,4),(14,5),(14,6),(2,10),(2,11),(2,12),(7,10),(7,11),(7,12),(9,10),(9,11),(9,12),(14,10),(14,11),(14,12),(4,2),(5,2),(6,2),(4,7),(5,7),(6,7),(4,9),(5,9),(6,9),(4,14),(5,14),(6,14),(10,2),(11,2),(12,2),(10,7),(11,7),(12,7),(10,9),(11,9),(12,9),(10,14),(11,14),(12,14)]),
        "floraison" : ((40,40), [(19,18),(19,19),(19,20),(20,17),(20,19),(20,21),(21,18),(21,19),(21,20)]),
        "block_switch_engine" : ((400,400), [(201,202),(201,203),(202,202),(202,203),(211,203),(212,204),(212,202),(214,204),(214,201),(215,201),(215,202),(216,201)]),
        "u" : ((200,200), [(101,101),(102,102),(103,102),(103,101),(104,103),(105,103),(105,102),(105,101),(105,105),(103,105),(102,105),(101,105),(101,104)]),
        "flat" : ((200,400), [(80,200),(81,200),(82,200),(83,200),(84,200),(85,200),(86,200),(87,200), (89,200),(90,200),(91,200),(92,200),(93,200),(97,200),(98,200),(99,200),(106,200),(107,200),(108,200),(109,200),(110,200),(111,200),(112,200),(114,200),(115,200),(116,200),(117,200),(118,200)])
    }
    choice = 'glider'

    color = 0 if rank == 0 else 1 
    sub_comm = globCom.Split(color, rank)
    size = sub_comm.size

    if len(sys.argv) > 1 :
        choice = sys.argv[1]

    resx = 800
    resy = 800
    if len(sys.argv) > 3 :
        resx = int(sys.argv[2])
        resy = int(sys.argv[3])
        
    print(f"Pattern initial choisi : {choice}")
    print(f"résolution écran : {resx,resy}")
    try:
        init_pattern = dico_patterns[choice]
    except KeyError:
        print("No such pattern. Available ones are:", dico_patterns.keys())
        exit(1)

    sub_size = sub_comm.size
    sub_rang = sub_comm.rank

    # ------------------------ calculateurs ------------------------
    #processus de rang != de 0
    if color != 0:
        #création du morceau de grille
        grid = Grille(sub_rang, sub_size, *init_pattern)
        next_rank = (sub_rang - 1) % sub_size
        prev_rank = (sub_rang + 1) % sub_size
    # ----------------- AFFICHAGE ----------------------------------
    #processus de rang 0
    else: 
        grid = Grille(0, 1, *init_pattern) #création d'une grille
        appli = App((resx, resy), grid)


    mustContinue = True
    while mustContinue:
        
        if color != 0:
        # ------------------------ calculateurs ------------------------
            # Echange de messages pour mettre à jour les cellules fantômes 
            # Envoie de la ligne 1 au voisin du haut, reçoit dans ligne 0
            t1 = time.time()
            req1 = sub_comm.Irecv(grid.cells[0, :], source=next_rank)
            req2 = sub_comm.Irecv(grid.cells[-1, :], source=prev_rank)
            
            # Envoie de la dernière ligne au voisin du bas, reçoit dans ligne -1
            sub_comm.Send(grid.cells[-2, :], dest=prev_rank)
            sub_comm.Send(grid.cells[1, :], dest=next_rank)
            
            req1.wait()
            req2.wait()
            
            diff = grid.compute_next_iteration()
            globCom.send(diff, dest=0, tag=7)
            t2 = time.time()
            print(f"temps de calcul de la nouvelle génération avec {nbp} processeurs : {t2-t1} s")

        else:
            # ----------------- AFFICHAGE -------------------------------
            t3 = time.time()
            for i in range(1, nbp): # Le processus de rang 0 reçoit des autres processus les changements d'état des cellules 
                diff_partiel = globCom.recv(source=i, tag=7)
                grid.modify(diff_partiel)

            appli.draw()
            t4 = time.time()
            print(f"temps d'affichage avec {nbp} processeurs : {t4-t3} s")
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    mustContinue = False
                    globCom.Abort()
