"""Deterministic wheat field shared by the stereo simulator and the 3D view."""
import numpy as np


FARM_XS, FARM_YS = 24., 16.
FARM_START = (-21., 0., 0.)
FARM_GOAL = (20.5, 0.)
ROW_SPACING = 2.4
ROWS = tuple((i+.5)*ROW_SPACING for i in range(-6,6))
PLANT_LIMIT = 18.5


def heights():
    x,y=np.meshgrid(np.linspace(-FARM_XS,FARM_XS,257),
                    np.linspace(-FARM_YS,FARM_YS,193))
    hills=.58*np.exp(-((x+9)/7.5)**2-((y-8)/4.3)**2)
    hills+=.64*np.exp(-((x-9)/7.)**2-((y+6)/4.)**2)
    hills-=.57*np.exp(-((x-8)/8.)**2-((y-9)/4.6)**2)
    hills-=.62*np.exp(-((x+12)/8.)**2-((y+9)/3.8)**2)
    hills+=.042*np.sin(.44*x+.7*y)*np.cos(.39*y)
    row=(y%ROW_SPACING)-ROW_SPACING/2
    end=np.clip((20.-np.abs(x))/1.5,0.,1.)
    shoulder=np.clip((15.-np.abs(y))/1.7,0.,1.)
    ridges=.21*np.exp(-(row/.36)**2)*end*shoulder
    z=hills+ridges
    return 2*(z-z.min())/(z.max()-z.min())-1.


def vegetation(height_at):
    rng=np.random.default_rng(5125)
    plants=[]
    for row in ROWS:
        for base in np.arange(-PLANT_LIMIT,PLANT_LIMIT,.38):
            x=base+rng.uniform(-.13,.13)
            y=row+rng.uniform(-.17,.17)
            plants.append((x,y,height_at(x,y),rng.uniform(.37,.69),rng.uniform(.12,.22),0))
    for x,y in rng.uniform([-22,-15],[22,15],size=(155,2)):
        plants.append((x,y,height_at(x,y),rng.uniform(.10,.26),rng.uniform(.12,.25),1))
    return plants


def scattered_rocks():
    rng=np.random.default_rng(914)
    rocks=[(5.0,.80,.58,.42,.54,.3)]
    while len(rocks)<23:
        x,y=rng.uniform([-19.,-14.],[19.,14.])
        if abs(y)<1.8 or any(np.hypot(x-a,y-b)<1.5 for a,b,*_ in rocks):
            continue
        rx,ry,h=rng.uniform([.20,.16,.28],[.75,.60,.82])
        rocks.append((x,y,rx,ry,h,rng.uniform(-np.pi,np.pi)))
    return rocks


def plant_mesh(plants,kind):
    """Two cross-sections per clump give both stereo eyes parallax."""
    vertices,faces=[],[]
    for x,y,z,h,width,p in plants:
        if p!=kind:
            continue
        for dx,dy in ((width,.025),(.025,width)):
            i=len(vertices)
            vertices.extend(((x-dx*.26,y-dy*.26,z),(x+dx*.26,y+dy*.26,z),
                             (x+dx*.32,y+dy*.32,z+h*.75),(x,y,z+h),
                             (x-dx*.32,y-dy*.32,z+h*.73),
                             (x-dx,y-dy,z+h*.47),(x+dx,y+dy,z+h*.54)))
            faces.extend(((i,i+1,i+2),(i,i+2,i+4),(i+2,i+3,i+4),
                          (i,i+4,i+5),(i+1,i+6,i+2)))
    return np.asarray(vertices),np.asarray(faces)
