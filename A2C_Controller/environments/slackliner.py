import numpy as np
from numpy import sin, cos, pi
import gym
from gym import core, spaces
from gym.utils import seeding

class SlacklineEnv(core.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 15
    }

    MAX_VEL = pi
    AVAIL_TORQUE = np.array([(x,y) for x in np.linspace(-20, 20, 10) for y in np.linspace(-20, 20, 10)])
    actions_num = 10

    # state: [xb  yb  phib  phit  phia  dxb  dyb  dphib  dphit  dphia]
    def __init__(self):
        self.viewer = None

        #general mechanical constants:
        self.g=9.81

        #segment properties:
        self.mb, self.mt, self.ma = 17, 51, 4
        self.lb, self.lt, self.la = .8, .5, .5
        self.sb, self.st, self.sa = self.lb/2, self.lt/2, self.la/2
        self.Jb, self.Jt, self.Ja = 1/12*self.mb*(self.lb)**2, 1/12*self.mt*(self.lt)**2, 1/12*self.ma*(self.la)**2

        #slackline properties:
        self.sag = .3
        self.c = (self.mt+self.mb+self.ma)*self.g/self.sag
        self.d = 2*.5*np.sqrt(self.c*(self.mt+self.mb+self.ma))

        self.state = self.reset()

    @property
    def observation_space(self):
        high = np.array([0.9, 0.9, pi, pi, pi, pi, pi, pi, pi, pi])
        low = -high
        observation_space = spaces.Box(low=low, high=high, dtype=np.float64)
        return observation_space

    @property
    def action_space(self):
        return spaces.Discrete(100)

    def reset(self):
        phib0 = 0
        phit0 = 0
        phia0 = 0
        r0 = self.sag
        delta0 = pi/500
        xb0 = r0*sin(delta0)+self.sb*sin(phib0)
        yb0 = -r0*cos(delta0)+self.sb*cos(phib0)
        self.state = np.array([xb0, yb0, phib0, phit0, phia0, 0, 0, 0, 0, 0])
        self.total_time = 0
        return self.state

    def step(self, a):
        dt = .01
        tend = 4
        s = self.state
        tau = self.AVAIL_TORQUE[a].flatten()
        tauH = tau[0]
        tauS = tau[1]

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, tauH)
        s_augmented = np.append(s_augmented, tauS)

        ns = rk4(self._dsdt, s_augmented, [0, dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:10]

        ns[0] = wrap(ns[0], -3*self.sag, 3*self.sag)
        ns[1] = wrap(ns[1], -3*self.sag, 3*self.sag)
        ns[2] = wrap(ns[2], -pi, pi)
        ns[3] = wrap(ns[3], -pi, pi)
        ns[4] = wrap(ns[4], -pi, pi)

        ns[5] = bound(ns[5], -self.MAX_VEL, self.MAX_VEL)
        ns[6] = bound(ns[6], -self.MAX_VEL, self.MAX_VEL)
        ns[7] = bound(ns[7], -self.MAX_VEL, self.MAX_VEL)
        ns[8] = bound(ns[8], -self.MAX_VEL, self.MAX_VEL)
        ns[9] = bound(ns[9], -self.MAX_VEL, self.MAX_VEL)
        self.state = ns
        reward = -np.sum(self.state[2:4]**2)-0.25*np.sum(self.state[5:]**2)

        self.total_time += dt
        terminal = self.total_time >= tend

        return (self.state, reward, terminal, {})

    def _dsdt(self, s_augmented, t):

        #general mechanical constants:
        self.g=9.81
        g=9.81

        #segment properties:
        self.mb, self.mt, self.ma = 17, 51, 4
        mb, mt, ma = 17, 51, 4
        self.lb, self.lt, self.la = .8, .5, .5
        lb, lt, la = .8, .5, .5
        self.sb, self.st, self.sa = lb/2, lt/2, la/2
        sb, st, sa = lb/2, lt/2, la/2
        self.Jb, self.Jt, self.Ja = 1/12*mb*(lb)**2, 1/12*mt*(lt)**2, 1/12*ma*(la)**2
        Jb, Jt, Ja = 1/12*mb*(lb)**2, 1/12*mt*(lt)**2, 1/12*ma*(la)**2

        #slackline properties:
        self.sag = .3
        sag = .3
        self.c = (mt+mb+ma)*g/sag
        c = (mt+mb+ma)*g/sag
        self.d = 2*.5*np.sqrt(c*(mt+mb+ma))
        d = 2*.5*np.sqrt(c*(mt+mb+ma))

        #state vector:
        #xb,yb,phib,phit,phia,dxb,dyb,dphib,dphit,dphia, 
        state = s_augmented
        xb=state[0]  #com position leg in x direction (m)
        yb=state[1]  #com position leg in y direction (m)
        phib = state[2]  #leg angle (rad)
        phit = state[3]  #trunk angle (rad)
        phia = state[4]  #arm angle (rad)
        #derivatives:
        dxb=state[5]  
        dyb=state[6]  
        dphib=state[7]  
        dphit=state[8]  
        dphia=state[9]  

        #All variables and parameters must be expressed in SI units (see above). 

        #calculate foot position and velocity (needed for slackline force):
        xf=xb-sb*sin(phib) #foot position in x direction
        yf=yb-sb*cos(phib) #foot position in y direction
        dxf=dxb-sb*dphib*cos(phib) #foot velocity in x-direction
        dyf=dyb+sb*dphib*sin(phib) #foot velocity in y-direction
    
        #calculate slackline geometry and force:    
        r=np.sqrt(xf**2+yf**2) #m, slackline radius
        drdt=(xf*dxf+yf*dyf)/np.sqrt(xf**2+yf**2) #radius derivative wrt time
        delta=np.arctan2(xf,-yf) #slackline angle
        Fs=r*c+drdt*d #force in radial direction
        FFx=-Fs*sin(delta) #force x-component acting on foot
        FFy=Fs*cos(delta) #force y-component acting on foot

        tauH = state[10]
        tauS = state[11]

        #mass matrix:
        massmatrix = np.array([[ ma + mb + mt, 0, lb*ma*cos(phib) + lb*mt*cos(phib) - ma*sb*cos(phib) - mt*sb*cos(phib), lt*ma*cos(phit) + mt*st*cos(phit), ma*sa*cos(phia)], \
            [0, ma + mb + mt,  ma*sb*sin(phib) - lb*mt*sin(phib) - lb*ma*sin(phib) + mt*sb*sin(phib), - lt*ma*sin(phit) - mt*st*sin(phit), -ma*sa*sin(phia)], \
            [(ma*cos(phib)*(2*lb - 2*sb))/2 + (mt*cos(phib)*(2*lb - 2*sb))/2, - (ma*sin(phib)*(2*lb - 2*sb))/2 - (mt*sin(phib)*(2*lb - 2*sb))/2, Jb + (ma*(sin(phib)*(lb*sin(phib) - sb*sin(phib))*(2*lb - 2*sb) + cos(phib)*(lb*cos(phib) - sb*cos(phib))*(2*lb - 2*sb)))/2 + (mt*(sin(phib)*(lb*sin(phib) - sb*sin(phib))*(2*lb - 2*sb) + cos(phib)*(lb*cos(phib) - sb*cos(phib))*(2*lb - 2*sb)))/2, (ma*(lt*cos(phib)*cos(phit)*(2*lb - 2*sb) + lt*sin(phib)*sin(phit)*(2*lb - 2*sb)))/2 + (mt*(st*cos(phib)*cos(phit)*(2*lb - 2*sb) + st*sin(phib)*sin(phit)*(2*lb - 2*sb)))/2, (ma*(sa*cos(phia)*cos(phib)*(2*lb - 2*sb) + sa*sin(phia)*sin(phib)*(2*lb - 2*sb)))/2], \
            [lt*ma*cos(phit) + mt*st*cos(phit), - lt*ma*sin(phit) - mt*st*sin(phit), lb*lt*ma*cos(phib - phit) - lt*ma*sb*cos(phib - phit) + lb*mt*st*cos(phib - phit) - mt*sb*st*cos(phib - phit), ma*lt**2 + mt*st**2 + Jt, lt*ma*sa*cos(phia - phit)], \
            [ma*sa*cos(phia), -ma*sa*sin(phia), lb*ma*sa*cos(phia - phib) - ma*sa*sb*cos(phia - phib),   lt*ma*sa*cos(phia - phit),   ma*sa**2 + Ja]])

        #further terms (gravity etc.):
        restterms =  np.array([[ - ma*sa*sin(phia)*dphia**2 - dphit*(dphit*lt*ma*sin(phit) + dphit*mt*st*sin(phit)) - dphib*(dphib*lb*ma*sin(phib) + dphib*lb*mt*sin(phib) - dphib*ma*sb*sin(phib) - dphib*mt*sb*sin(phib))], \
            [- ma*sa*cos(phia)*dphia**2 - dphit*(dphit*lt*ma*cos(phit) + dphit*mt*st*cos(phit)) + g*(ma + mb + mt) - dphib*(dphib*lb*ma*cos(phib) + dphib*lb*mt*cos(phib) - dphib*ma*sb*cos(phib) - dphib*mt*sb*cos(phib))], \
            [(lb - sb)*(dphib*dyb*ma*cos(phib) - g*mt*sin(phib) - g*ma*sin(phib) + dphib*dyb*mt*cos(phib) + dphib*dxb*ma*sin(phib) + dphib*dxb*mt*sin(phib) + dphib*dphit*lt*ma*sin(phib - phit) - dphia*dphib*ma*sa*sin(phia - phib) + dphib*dphit*mt*st*sin(phib - phit)) - dphib*((ma*(sin(phib)*(2*lb - 2*sb)*(dxb + dphib*lb*cos(phib) + dphit*lt*cos(phit) + dphia*sa*cos(phia) - dphib*sb*cos(phib)) - sin(phib)*(dphib*lb*cos(phib) - dphib*sb*cos(phib))*(2*lb - 2*sb) - cos(phib)*(2*lb - 2*sb)*(dphib*lb*sin(phib) - dyb + dphit*lt*sin(phit) + dphia*sa*sin(phia) - dphib*sb*sin(phib)) + cos(phib)*(dphib*lb*sin(phib) - dphib*sb*sin(phib))*(2*lb - 2*sb)))/2 + (mt*(sin(phib)*(2*lb - 2*sb)*(dxb + dphib*lb*cos(phib) - dphib*sb*cos(phib) + dphit*st*cos(phit)) + cos(phib)*(2*lb - 2*sb)*(dyb - dphib*lb*sin(phib) + dphib*sb*sin(phib) - dphit*st*sin(phit)) - sin(phib)*(dphib*lb*cos(phib) - dphib*sb*cos(phib))*(2*lb - 2*sb) + cos(phib)*(dphib*lb*sin(phib) - dphib*sb*sin(phib))*(2*lb - 2*sb)))/2) - dphit*((mt*(dphit*st*cos(phib)*sin(phit)*(2*lb - 2*sb) - dphit*st*cos(phit)*sin(phib)*(2*lb - 2*sb)))/2 + (ma*(dphit*lt*cos(phib)*sin(phit)*(2*lb - 2*sb) - dphit*lt*cos(phit)*sin(phib)*(2*lb - 2*sb)))/2) + (dphia*ma*(dphia*sa*cos(phia)*sin(phib)*(2*lb - 2*sb) - dphia*sa*cos(phib)*sin(phia)*(2*lb - 2*sb)))/2], \
            [dphit*dyb*lt*ma*cos(phit) - dphit*(dyb*lt*ma*cos(phit) + dyb*mt*st*cos(phit) + dxb*lt*ma*sin(phit) + dxb*mt*st*sin(phit) - dphib*lb*lt*ma*sin(phib - phit) - dphia*lt*ma*sa*sin(phia - phit) + dphib*lt*ma*sb*sin(phib - phit) - dphib*lb*mt*st*sin(phib - phit) + dphib*mt*sb*st*sin(phib - phit)) - g*lt*ma*sin(phit) - g*mt*st*sin(phit) - dphia**2*lt*ma*sa*sin(phia - phit) - dphib*(dphib*lb*lt*ma*sin(phib - phit) - dphib*lt*ma*sb*sin(phib - phit) + dphib*lb*mt*st*sin(phib - phit) - dphib*mt*sb*st*sin(phib - phit)) + dphit*dyb*mt*st*cos(phit) + dphit*dxb*lt*ma*sin(phit) + dphit*dxb*mt*st*sin(phit) - dphib*dphit*lb*lt*ma*sin(phib - phit) - dphia*dphit*lt*ma*sa*sin(phia - phit) + dphib*dphit*lt*ma*sb*sin(phib - phit) - dphib*dphit*lb*mt*st*sin(phib - phit) + dphib*dphit*mt*sb*st*sin(phib - phit)], \
            [dphib*(dphib*lb*ma*sa*sin(phia - phib) - dphib*ma*sa*sb*sin(phia - phib)) - dphia*(dyb*ma*sa*cos(phia) + dxb*ma*sa*sin(phia) + dphib*lb*ma*sa*sin(phia - phib) + dphit*lt*ma*sa*sin(phia - phit) - dphib*ma*sa*sb*sin(phia - phib)) + ma*sa*(dphia*dxb*sin(phia) - g*sin(phia) + dphia*dyb*cos(phia) + dphia*dphib*lb*sin(phia - phib) + dphia*dphit*lt*sin(phia - phit) - dphia*dphib*sb*sin(phia - phib)) + dphit**2*lt*ma*sa*sin(phia - phit)]])

        #generalized applied forces, from slackline and joint moments:
        appliedforces = np.array([[FFx], [FFy], [tauH-FFx*sb*cos(phib)+FFy*sb*sin(phib)], [-tauH+tauS], [-tauS]])

        #solve equations of motion for accelerations:
        accelerations = np.linalg.inv(massmatrix) @ (appliedforces-restterms)

        statederivative = state[5:10]
        statederivative = np.append(statederivative, accelerations.flatten())
        statederivative = np.append(statederivative, np.array([0,0]))
        return statederivative

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            bound = self.lb + self.lt + self.la + 0.5  
            self.viewer.set_bounds(-bound,bound,-0.5,bound)

        if s is None: return None
        #xb,yb,phib,phit,phia,dxb,dyb,dphib,dphit,dphia,

        p2 = [self.lb*cos(s[2]), self.lb*sin(s[2])]
        p3 = [p2[0]+self.lt*cos(s[3]), p2[1]+self.lt*sin(s[3])]
        p4 = [p3[0]+self.la*cos(s[4]), p3[1]+self.la*sin(s[4])]

        xys = np.array([[0,0], p2, p3, p4])[:,::-1]
        thetas = [pi/2-s[2], pi/2-s[3], pi/2-s[4]]
        link_lengths = [self.lb, self.lt, self.la]

        # self.viewer.draw_line((0, p1[1]), (0, p1[0]))
        for ((x,y),th,llen) in zip(xys, thetas, link_lengths):
            l,r,t,b = 0, llen, .03, -.03
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0,.8, .8)
            circ = self.viewer.draw_circle(.03)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def wrap(x, m, M):
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

def bound(x, m, M=None):
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def rk4(derivs, y0, t, *args, **kwargs):
    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout