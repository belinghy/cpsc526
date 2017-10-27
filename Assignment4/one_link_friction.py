from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys

#  from pyquaternion import Quaternion    ## would be useful for 3D simulation
import numpy as np
import matplotlib.pyplot as plt

window = 0     # number of the glut window
theta = 0
simTime = 0
dT = 0.02
simRun = True
RAD_TO_DEG = 180.0/3.14160
init_angle = 3.1416 / 4
k_d = 0.2

plt.ion()

#####################################################
#### Link class, i.e., for a rigid body
#####################################################

class Link:
    color=[0,0,0]    ## draw color
    size=[1,1,1]     ## dimensions
    mass = 1.0       ## mass in kg
    izz = 1.0        ## moment of inertia about z-axis
    theta=0          ## 2D orientation  (will need to change for 3D)
    omega=0          ## 2D angular velocity
    posn=np.array([0.0,0.0,0.0])     ## 3D position (keep z=0 for 2D)
    vel=np.array([0.0,0.0,0.0])      ## initial velocity
    def draw(self):      ### steps to draw a link
        glPushMatrix()                                            ## save copy of coord frame
        glTranslatef(self.posn[0], self.posn[1], self.posn[2])    ## move 
        glRotatef(self.theta*RAD_TO_DEG,  0,0,1)                             ## rotate
        glScale(self.size[0], self.size[1], self.size[2])         ## set size
        glColor3f(self.color[0], self.color[1], self.color[2])    ## set colour
        DrawCube()                                                ## draw a scaled cube
        glPopMatrix()                                             ## restore old coord frame

#####################################################
#### main():   launches app
#####################################################

def main():
    global window
    global link1       
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)     # display mode 
    glutInitWindowSize(640, 480)                                  # window size
    glutInitWindowPosition(0, 0)                                  # window coords for mouse start at top-left
    window = glutCreateWindow("CPSC 526 Simulation Template")
    glutDisplayFunc(DrawWorld)       # register the function to draw the world
    # glutFullScreen()               # full screen
    glutIdleFunc(SimWorld)          # when doing nothing, redraw the scene
    glutReshapeFunc(ReSizeGLScene)   # register the function to call when window is resized
    glutKeyboardFunc(keyPressed)     # register the function to call when keyboard is pressed
    InitGL(640, 480)                 # initialize window
    
    link1 = Link();
    resetSim()
    
    glutMainLoop()                   # start event processing loop

#####################################################
#### keyPressed():  called whenever a key is pressed
#####################################################

def resetSim():
    global link1, theta
    global simTime, simRun

    printf("Simulation reset\n")
    simRun = True
    simTime = 0

    link1.size=[0.04, 1.0, 0.12]
    link1.color=[1,0.9,0.9]
    link1.posn=np.array([0.0,0.0,0.0])
    link1.vel=np.array([0.0,0.0,0.0])
    link1.theta = init_angle
    link1.omega = 0        ## radians per second
    link1.izz = 1.0 / 3

#####################################################
#### keyPressed():  called whenever a key is pressed
#####################################################

def keyPressed(key,x,y):
    global simRun
    ch = key.decode("utf-8")
    if ch == ' ':                #### toggle the simulation
        if (simRun == True):
            simRun = False
        else:
            simRun = True
    elif ch == chr(27):          #### ESC key
        sys.exit()
    elif ch == 'q':              #### quit
        sys.exit()
    elif ch == 'r':              #### reset simulation
        resetSim()

#####################################################
#### SimWorld():  simulates a time step
#####################################################

def SimWorld():
    global simTime, dT, simRun
    global link1

    if (simRun==False):             ## is simulation stopped?
        return

    #### solve for the equations of motion (simple in this case!)
    acc1 = np.array([0, -10, 0])       ### linear acceleration = [0, -G, 0]
    omega_dot1 = 0.0                 ### assume no angular acceleration

    ####  for the constrained one-link pendulum, and the 4-link pendulum,
    ####  you will want to build the equations of motion as a linear system, and then solve that.
    ####  Here is a simple example of using numpy to solve a linear system.
    m = link1.mass
    I = link1.izz
    rx = 1.0 * np.sin(link1.theta)/2
    ry = -1.0 * np.cos(link1.theta)/2
    z = 0.0
    k = -1.0
    g = 10.0
    w = link1.omega
    a = np.array([[m, z, z, z, z, z, k, z, z],
                  [z, m, z, z, z, z, z, k, z],
                  [z, z, m, z, z, z, z, z, k],
                  [z, z, z, I, z, z, z, z, ry],
                  [z, z, z, z, I, z, z, z, -rx],
                  [z, z, z, z, z, I, -ry, rx, z],
                  [k, z, z, z, z, -ry, z, z, z],
                  [z, k, z, z, z, rx, z, z, z],
                  [z, z, k, ry, -rx, z, z, z, z]])
    b = np.array([z, -g, z, z, z, -k_d*w, w*w*rx, w*w*ry, z])
    x = np.linalg.solve(a, b)
    #print(x)   # [ -2.17647059  53.54411765  56.63235294]
    acc1 = np.array([x[0], x[1], x[2]])
    omega_dot1 = x[5]
    # print(acc1)

    cpoint = link1.posn - np.array([rx, ry, 0])
    init_cpoint = np.array([-1.0 * np.sin(init_angle) / 2, 1 * np.cos(init_angle) / 2, 0])
    cvelocity = (link1.vel + np.cross(np.array([0, 0, link1.omega]), np.array([-rx, -ry, 0])))
    factor = 1*(cpoint - init_cpoint) + 10*(cvelocity)
    acc1 -= factor

    #### explicit Euler integration to update the state
    link1.posn += link1.vel*dT
    link1.vel += acc1*dT
    link1.theta += link1.omega*dT
    link1.omega += omega_dot1*dT

    # print("{} / {} / {}".format(link1.posn, link1.vel, acc1))
    KE = 0.5 * link1.izz * link1.omega * link1.omega
    PE = m * g * 1.0 * (1 - np.cos(link1.theta)) / 2
    TE = KE + PE

    # print("{}: {} / {} / {}".format(link1.theta*RAD_TO_DEG, KE, PE, TE))
    plt.scatter(simTime, KE, s=0.5, c="b", marker="o")
    plt.scatter(simTime, PE, s=0.5, c="g", marker="o")
    plt.scatter(simTime, TE, s=1.0, c="r", marker="x")
    plt.pause(dT)
    simTime += dT

    #### draw the updated state
    DrawWorld()
    printf("simTime=%.2f\n",simTime)

#####################################################
#### DrawWorld():  draw the world
#####################################################

def DrawWorld():
    global link1

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	# Clear The Screen And The Depth Buffer
    glLoadIdentity();
    gluLookAt(1,1,3,  0,0,0,  0,1,0)

    DrawOrigin()
    link1.draw()

    glutSwapBuffers()                      # swap the buffers to display what was just drawn

#####################################################
#### initGL():  does standard OpenGL initialization work
#####################################################

def InitGL(Width, Height):				# We call this right after our OpenGL window is created.
    glClearColor(1.0, 1.0, 0.9, 0.0)	# This Will Clear The Background Color To Black
    glClearDepth(1.0)					# Enables Clearing Of The Depth Buffer
    glDepthFunc(GL_LESS)				# The Type Of Depth Test To Do
    glEnable(GL_DEPTH_TEST)				# Enables Depth Testing
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);    glEnable( GL_LINE_SMOOTH );
    glShadeModel(GL_SMOOTH)				# Enables Smooth Color Shading
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()					# Reset The Projection Matrix
    gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

#####################################################
#### ReSizeGLScene():    called when window is resized
#####################################################

def ReSizeGLScene(Width, Height):
    if Height == 0:						# Prevent A Divide By Zero If The Window Is Too Small 
        Height = 1
    glViewport(0, 0, Width, Height)		# Reset The Current Viewport And Perspective Transformation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)    ## 45 deg horizontal field of view, aspect ratio, near, far
    glMatrixMode(GL_MODELVIEW)

#####################################################
#### DrawOrigin():  draws RGB lines for XYZ origin of coordinate system
#####################################################

def DrawOrigin():
    glLineWidth(3.0);

    glColor3f(1,0.0,0.0)   ## light red x-axis
    glBegin(GL_LINES)
    glVertex3f(0,0,0)
    glVertex3f(1,0,0)
    glEnd()

    glColor3f(0.0,1,0.0)   ## light green y-axis
    glBegin(GL_LINES)
    glVertex3f(0,0,0)
    glVertex3f(0,1,0)
    glEnd()

    glColor3f(0.0,0.0,1)   ## light blue z-axis
    glBegin(GL_LINES)
    glVertex3f(0,0,0)
    glVertex3f(0,0,1)
    glEnd()

#####################################################
#### DrawCube():  draws a cube that spans from (-1,-1,-1) to (1,1,1)
#####################################################

def DrawCube():

    glScalef(0.5,0.5,0.5);                  # dimensions below are for a 2x2x2 cube, so scale it down by a half first
    glBegin(GL_QUADS);			# Start Drawing The Cube

    glVertex3f( 1.0, 1.0,-1.0);		# Top Right Of The Quad (Top)
    glVertex3f(-1.0, 1.0,-1.0);		# Top Left Of The Quad (Top)
    glVertex3f(-1.0, 1.0, 1.0);		# Bottom Left Of The Quad (Top)
    glVertex3f( 1.0, 1.0, 1.0);		# Bottom Right Of The Quad (Top)

    glVertex3f( 1.0,-1.0, 1.0);		# Top Right Of The Quad (Bottom)
    glVertex3f(-1.0,-1.0, 1.0);		# Top Left Of The Quad (Bottom)
    glVertex3f(-1.0,-1.0,-1.0);		# Bottom Left Of The Quad (Bottom)
    glVertex3f( 1.0,-1.0,-1.0);		# Bottom Right Of The Quad (Bottom)

    glVertex3f( 1.0, 1.0, 1.0);		# Top Right Of The Quad (Front)
    glVertex3f(-1.0, 1.0, 1.0);		# Top Left Of The Quad (Front)
    glVertex3f(-1.0,-1.0, 1.0);		# Bottom Left Of The Quad (Front)
    glVertex3f( 1.0,-1.0, 1.0);		# Bottom Right Of The Quad (Front)

    glVertex3f( 1.0,-1.0,-1.0);		# Bottom Left Of The Quad (Back)
    glVertex3f(-1.0,-1.0,-1.0);		# Bottom Right Of The Quad (Back)
    glVertex3f(-1.0, 1.0,-1.0);		# Top Right Of The Quad (Back)
    glVertex3f( 1.0, 1.0,-1.0);		# Top Left Of The Quad (Back)

    glVertex3f(-1.0, 1.0, 1.0);		# Top Right Of The Quad (Left)
    glVertex3f(-1.0, 1.0,-1.0);		# Top Left Of The Quad (Left)
    glVertex3f(-1.0,-1.0,-1.0);		# Bottom Left Of The Quad (Left)
    glVertex3f(-1.0,-1.0, 1.0);		# Bottom Right Of The Quad (Left)

    glVertex3f( 1.0, 1.0,-1.0);		# Top Right Of The Quad (Right)
    glVertex3f( 1.0, 1.0, 1.0);		# Top Left Of The Quad (Right)
    glVertex3f( 1.0,-1.0, 1.0);		# Bottom Left Of The Quad (Right)
    glVertex3f( 1.0,-1.0,-1.0);		# Bottom Right Of The Quad (Right)
    glEnd();				# Done Drawing The Quad

    ### Draw the wireframe edges
    glColor3f(0.0, 0.0, 0.0);
    glLineWidth(1.0);
     
    glBegin(GL_LINE_LOOP);		
    glVertex3f( 1.0, 1.0,-1.0);		# Top Right Of The Quad (Top)
    glVertex3f(-1.0, 1.0,-1.0);		# Top Left Of The Quad (Top)
    glVertex3f(-1.0, 1.0, 1.0);		# Bottom Left Of The Quad (Top)
    glVertex3f( 1.0, 1.0, 1.0);		# Bottom Right Of The Quad (Top)
    glEnd();				# Done Drawing The Quad

    glBegin(GL_LINE_LOOP);		
    glVertex3f( 1.0,-1.0, 1.0);		# Top Right Of The Quad (Bottom)
    glVertex3f(-1.0,-1.0, 1.0);		# Top Left Of The Quad (Bottom)
    glVertex3f(-1.0,-1.0,-1.0);		# Bottom Left Of The Quad (Bottom)
    glVertex3f( 1.0,-1.0,-1.0);		# Bottom Right Of The Quad (Bottom)
    glEnd();				# Done Drawing The Quad

    glBegin(GL_LINE_LOOP);		
    glVertex3f( 1.0, 1.0, 1.0);		# Top Right Of The Quad (Front)
    glVertex3f(-1.0, 1.0, 1.0);		# Top Left Of The Quad (Front)
    glVertex3f(-1.0,-1.0, 1.0);		# Bottom Left Of The Quad (Front)
    glVertex3f( 1.0,-1.0, 1.0);		# Bottom Right Of The Quad (Front)
    glEnd();				# Done Drawing The Quad

    glBegin(GL_LINE_LOOP);		
    glVertex3f( 1.0,-1.0,-1.0);		# Bottom Left Of The Quad (Back)
    glVertex3f(-1.0,-1.0,-1.0);		# Bottom Right Of The Quad (Back)
    glVertex3f(-1.0, 1.0,-1.0);		# Top Right Of The Quad (Back)
    glVertex3f( 1.0, 1.0,-1.0);		# Top Left Of The Quad (Back)
    glEnd();				# Done Drawing The Quad

    glBegin(GL_LINE_LOOP);		
    glVertex3f(-1.0, 1.0, 1.0);		# Top Right Of The Quad (Left)
    glVertex3f(-1.0, 1.0,-1.0);		# Top Left Of The Quad (Left)
    glVertex3f(-1.0,-1.0,-1.0);		# Bottom Left Of The Quad (Left)
    glVertex3f(-1.0,-1.0, 1.0);		# Bottom Right Of The Quad (Left)
    glEnd();				# Done Drawing The Quad

    glBegin(GL_LINE_LOOP);		
    glVertex3f( 1.0, 1.0,-1.0);		# Top Right Of The Quad (Right)
    glVertex3f( 1.0, 1.0, 1.0);		# Top Left Of The Quad (Right)
    glVertex3f( 1.0,-1.0, 1.0);		# Bottom Left Of The Quad (Right)
    glVertex3f( 1.0,-1.0,-1.0);		# Bottom Right Of The Quad (Right)
    glEnd();				# Done Drawing The Quad

####################################################
# printf()  
####################################################

def printf(format, *args):
    sys.stdout.write(format % args)

################################################################################
# start the app

print ("Hit ESC key to quit.")
main()
        
