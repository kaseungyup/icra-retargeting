import math,time,os
import numpy as np
import tkinter as tk
import shapely as sp # handle polygon
from shapely import Polygon,LineString,Point # handle polygons
from scipy.spatial.distance import cdist

def rot_mtx(deg):
    """
        2 x 2 rotation matrix
    """
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

def pr2t(p,R):
    """ 
        Convert pose to transformation matrix 
    """
    p0 = p.ravel() # flatten
    T = np.block([
        [R, p0[:, np.newaxis]],
        [np.zeros(3), 1]
    ])
    return T

def t2pr(T):
    """
        T to p and R
    """   
    p = T[:3,3]
    R = T[:3,:3]
    return p,R

def t2p(T):
    """
        T to p 
    """   
    p = T[:3,3]
    return p

def t2r(T):
    """
        T to R
    """   
    R = T[:3,:3]
    return R    

def rpy2r(rpy_rad):
    """
        roll,pitch,yaw in radian to R
    """
    roll  = rpy_rad[0]
    pitch = rpy_rad[1]
    yaw   = rpy_rad[2]
    Cphi  = np.math.cos(roll)
    Sphi  = np.math.sin(roll)
    Cthe  = np.math.cos(pitch)
    Sthe  = np.math.sin(pitch)
    Cpsi  = np.math.cos(yaw)
    Spsi  = np.math.sin(yaw)
    R     = np.array([
        [Cpsi * Cthe, -Spsi * Cphi + Cpsi * Sthe * Sphi, Spsi * Sphi + Cpsi * Sthe * Cphi],
        [Spsi * Cthe, Cpsi * Cphi + Spsi * Sthe * Sphi, -Cpsi * Sphi + Spsi * Sthe * Cphi],
        [-Sthe, Cthe * Sphi, Cthe * Cphi]
    ])
    assert R.shape == (3, 3)
    return R

def r2rpy(R,unit='rad'):
    """
        Rotation matrix to roll,pitch,yaw in radian
    """
    roll  = math.atan2(R[2, 1], R[2, 2])
    pitch = math.atan2(-R[2, 0], (math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    yaw   = math.atan2(R[1, 0], R[0, 0])
    if unit == 'rad':
        out = np.array([roll, pitch, yaw])
    elif unit == 'deg':
        out = np.array([roll, pitch, yaw])*180/np.pi
    else:
        out = None
        raise Exception("[r2rpy] Unknown unit:[%s]"%(unit))
    return out    

def r2w(R):
    """
        R to \omega
    """
    el = np.array([
            [R[2,1] - R[1,2]],
            [R[0,2] - R[2,0]], 
            [R[1,0] - R[0,1]]
        ])
    norm_el = np.linalg.norm(el)
    if norm_el > 1e-10:
        w = np.arctan2(norm_el, np.trace(R)-1) / norm_el * el
    elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
        w = np.array([[0, 0, 0]]).T
    else:
        w = np.math.pi/2 * np.array([[R[0,0]+1], [R[1,1]+1], [R[2,2]+1]])
    return w.flatten()

def r2quat(R):
    """ 
        Convert Rotation Matrix to Quaternion.  See rotation.py for notes 
        (https://gist.github.com/machinaut/dab261b78ac19641e91c6490fb9faa96)
    """
    R = np.asarray(R, dtype=np.float64)
    Qxx, Qyx, Qzx = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    Qxy, Qyy, Qzy = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    Qxz, Qyz, Qzz = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(R.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=['multi_index'])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q

def skew(x):
    """ 
        Get a skew-symmetric matrix
    """
    x_hat = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
    return x_hat

def rodrigues(a=np.array([1,0,0]),q_rad=0.0):
    """
        Compute the rotation matrix from an angular velocity vector
    """
    a_norm = np.linalg.norm(a)
    if abs(a_norm-1) > 1e-6:
        print ("[rodrigues] norm of a should be 1.0 not [%.2e]."%(a_norm))
        return np.eye(3)
    
    a = a / a_norm
    q_rad = q_rad * a_norm
    a_hat = skew(a)
    
    R = np.eye(3) + a_hat*np.sin(q_rad) + a_hat@a_hat*(1-np.cos(q_rad))
    return R
    
def np_uv(vec):
    """
        Get unit vector
    """
    x = np.array(vec)
    return x/np.linalg.norm(x)

def get_rotation_matrix_from_two_points(p_fr,p_to):
    p_a  = np.copy(np.array([0,0,1]))
    if np.linalg.norm(p_to-p_fr) < 1e-8: # if two points are too close
        return np.eye(3)
    p_b  = (p_to-p_fr)/np.linalg.norm(p_to-p_fr)
    v    = np.cross(p_a,p_b)
    S = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    if np.linalg.norm(v) == 0:
        R = np.eye(3,3)
    else:
        R = np.eye(3,3) + S + S@S*(1-np.dot(p_a,p_b))/(np.linalg.norm(v)*np.linalg.norm(v))
    return R
    

def trim_scale(x,th):
    """
        Trim scale
    """
    x         = np.copy(x)
    x_abs_max = np.abs(x).max()
    if x_abs_max > th:
        x = x*th/x_abs_max
    return x

def soft_squash(x,x_min=-1,x_max=+1,margin=0.1):
    """
        Soft squashing numpy array
    """
    def th(z,m=0.0):
        # thresholding function 
        return (m)*(np.exp(2/m*z)-1)/(np.exp(2/m*z)+1)
    x_in = np.copy(x)
    idxs_upper = np.where(x_in>(x_max-margin))
    x_in[idxs_upper] = th(x_in[idxs_upper]-(x_max-margin),m=margin) + (x_max-margin)
    idxs_lower = np.where(x_in<(x_min+margin))
    x_in[idxs_lower] = th(x_in[idxs_lower]-(x_min+margin),m=margin) + (x_min+margin)
    return x_in    

def soft_squash_multidim(
    x      = np.random.randn(100,5),
    x_min  = -np.ones(5),
    x_max  = np.ones(5),
    margin = 0.1):
    """
        Multi-dim version of 'soft_squash' function
    """
    x_squash = np.copy(x)
    dim      = x.shape[1]
    for d_idx in range(dim):
        x_squash[:,d_idx] = soft_squash(
            x=x[:,d_idx],x_min=x_min[d_idx],x_max=x_max[d_idx],margin=margin)
    return x_squash 

def kernel_se(X1,X2,hyp={'g':1,'l':1}):
    """
        Squared exponential (SE) kernel function
    """
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    return K

def kernel_levse(X1,X2,L1,L2,hyp={'g':1,'l':1}):
    """
        Leveraged SE kernel function
    """
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    L = np.cos(np.pi/2.0*cdist(L1,L2,'cityblock'))
    return np.multiply(K,L)

def is_point_in_polygon(point,polygon):
    """
        Is the point inside the polygon
    """
    if isinstance(point,np.ndarray):
        point_check = Point(point)
    else:
        point_check = point
    return sp.contains(polygon,point_check)

def is_point_feasible(point,obs_list):
    """
        Is the point feasible w.r.t. obstacle list
    """
    result = is_point_in_polygon(point,obs_list) # is the point inside each obstacle?
    if sum(result) == 0:
        return True
    else:
        return False

def is_point_to_point_connectable(point1,point2,obs_list):
    """
        Is the line connecting two points connectable
    """
    result = sp.intersects(LineString([point1,point2]),obs_list)
    if sum(result) == 0:
        return True
    else:
        return False
    
class TicTocClass(object):
    """
        Tic toc
        tictoc = TicTocClass()
        tictoc.tic()
        ~~
        tictoc.toc()
    """
    def __init__(self,name='tictoc',print_every=1):
        """
            Initialize
        """
        self.name         = name
        self.time_start   = time.time()
        self.time_end     = time.time()
        self.print_every  = print_every
        self.time_elapsed = 0.0

    def tic(self):
        """
            Tic
        """
        self.time_start = time.time()

    def toc(self,str=None,cnt=0,VERBOSE=True,RETURN=False):
        """
            Toc
        """
        self.time_end = time.time()
        self.time_elapsed = self.time_end - self.time_start
        if VERBOSE:
            if self.time_elapsed <1.0:
                time_show = self.time_elapsed*1000.0
                time_unit = 'ms'
            elif self.time_elapsed <60.0:
                time_show = self.time_elapsed
                time_unit = 's'
            else:
                time_show = self.time_elapsed/60.0
                time_unit = 'min'
            if (cnt % self.print_every) == 0:
                if str is None:
                    print ("%s Elapsed time:[%.2f]%s"%
                        (self.name,time_show,time_unit))
                else:
                    print ("%s Elapsed time:[%.2f]%s"%
                        (str,time_show,time_unit))
        if RETURN:
            return self.time_elapsed

def get_interp_const_vel_traj(traj_anchor,vel=1.0,HZ=100,ord=np.inf):
    """
        Get linearly interpolated constant velocity trajectory
    """
    L = traj_anchor.shape[0]
    D = traj_anchor.shape[1]
    dists = np.zeros(L)
    for tick in range(L):
        if tick > 0:
            p_prev,p_curr = traj_anchor[tick-1,:],traj_anchor[tick,:]
            dists[tick] = np.linalg.norm(p_prev-p_curr,ord=ord)
    times_anchor = np.cumsum(dists/vel) # [L]
    L_interp = int(times_anchor[-1]*HZ)
    times_interp = np.linspace(0,times_anchor[-1],L_interp) # [L_interp]
    traj_interp = np.zeros((L_interp,D)) # [L_interp x D]
    for d_idx in range(D):
        traj_interp[:,d_idx] = np.interp(times_interp,times_anchor,traj_anchor[:,d_idx])
    return times_interp,traj_interp

def meters2xyz(depth_img,cam_matrix):
    """
        Scaled depth image to pointcloud
    """
    fx = cam_matrix[0][0]
    cx = cam_matrix[0][2]
    fy = cam_matrix[1][1]
    cy = cam_matrix[1][2]
    
    height = depth_img.shape[0]
    width = depth_img.shape[1]
    indices = np.indices((height, width),dtype=np.float32).transpose(1,2,0)
    
    z_e = depth_img
    x_e = (indices[..., 1] - cx) * z_e / fx
    y_e = (indices[..., 0] - cy) * z_e / fy
    
    # Order of y_ e is reversed !
    xyz_img = np.stack([z_e, -x_e, -y_e], axis=-1) # [H x W x 3] 
    return xyz_img # [H x W x 3]

def compute_view_params(camera_pos,target_pos,up_vector=np.array([0,0,1])):
    """Compute azimuth, distance, elevation, and lookat for a viewer given camera pose in 3D space.

    Args:
        camera_pos (np.ndarray): 3D array of camera position.
        target_pos (np.ndarray): 3D array of target position.
        up_vector (np.ndarray): 3D array of up vector.

    Returns:
        tuple: Tuple containing azimuth, distance, elevation, and lookat values.
    """
    # Compute camera-to-target vector and distance
    cam_to_target = target_pos - camera_pos
    distance = np.linalg.norm(cam_to_target)

    # Compute azimuth and elevation
    azimuth = np.arctan2(cam_to_target[1], cam_to_target[0])
    azimuth = np.rad2deg(azimuth) # [deg]
    elevation = np.arcsin(cam_to_target[2] / distance)
    elevation = np.rad2deg(elevation) # [deg]

    # Compute lookat point
    lookat = target_pos

    # Compute camera orientation matrix
    zaxis = cam_to_target / distance
    xaxis = np.cross(up_vector, zaxis)
    yaxis = np.cross(zaxis, xaxis)
    cam_orient = np.array([xaxis, yaxis, zaxis])

    # Return computed values
    return azimuth, distance, elevation, lookat

def sample_xyzs(n_sample,x_range=[0,1],y_range=[0,1],z_range=[0,1],min_dist=0.1,xy_margin=0.0):
    """
        Sample a point in three dimensional space with the minimum distance between points
    """
    xyzs = np.zeros((n_sample,3))
    for p_idx in range(n_sample):
        while True:
            x_rand = np.random.uniform(low=x_range[0]+xy_margin,high=x_range[1]-xy_margin)
            y_rand = np.random.uniform(low=y_range[0]+xy_margin,high=y_range[1]-xy_margin)
            z_rand = np.random.uniform(low=z_range[0],high=z_range[1])
            xyz = np.array([x_rand,y_rand,z_rand])
            if p_idx == 0: break
            devc = cdist(xyz.reshape((-1,3)),xyzs[:p_idx,:].reshape((-1,3)),'euclidean')
            if devc.min() > min_dist: break # minimum distance between objects
        xyzs[p_idx,:] = xyz
    return xyzs

def create_folder_if_not_exists(file_path):
    """ 
        Create folder if not exist
    """
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print ("[%s] created."%(folder_path))
        
class MultiSliderClass(object):
    """
        GUI with multiple sliders
    """
    def __init__(self,
                 n_slider      = 10,
                 title         = 'Multiple Sliders',
                 window_width  = 500,
                 window_height = None,
                 x_offset      = 500,
                 y_offset      = 100,
                 slider_width  = 400,
                 label_texts   = None,
                 slider_mins   = None,
                 slider_maxs   = None,
                 slider_vals   = None,
                 resolution    = 0.1,
                 VERBOSE       = True
        ):
        """
            Initialze multiple sliders
        """
        self.n_slider      = n_slider
        self.title         = title
        
        self.window_width  = window_width
        if window_height is None:
            self.window_height = self.n_slider*40
        else:
            self.window_height = window_height
        self.x_offset      = x_offset
        self.y_offset      = y_offset
        self.slider_width  = slider_width
        
        self.resolution    = resolution
        self.VERBOSE       = VERBOSE
        
        # Slider values
        self.slider_values = np.zeros(self.n_slider)
        
        # Initial/default slider settings
        self.label_texts   = label_texts
        self.slider_mins   = slider_mins
        self.slider_maxs   = slider_maxs
        self.slider_vals   = slider_vals
        
        # Create main window
        self.gui = tk.Tk()
        self.gui.title("%s"%(self.title))
        self.gui.geometry("%dx%d+%d+%d"%
                          (self.window_width,self.window_height,self.x_offset,self.y_offset))
        
        # Create vertical scrollbar
        self.scrollbar = tk.Scrollbar(self.gui,orient=tk.VERTICAL)
        
        # Create a Canvas widget with the scrollbar attached
        self.canvas = tk.Canvas(self.gui,yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure the scrollbar to control the canvas
        self.scrollbar.config(command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create a frame inside the canvas to hold the sliders
        self.sliders_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0,0),window=self.sliders_frame,anchor=tk.NW)
        
        # Create sliders
        self.sliders = self.create_sliders()
        
        # Update the canvas scroll region when the sliders_frame changes size
        self.sliders_frame.bind("<Configure>",self.cb_scroll)
        
    def cb_scroll(self,event):    
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def cb_slider(self,slider_idx,slider_value):
        """
            Slider callback function
        """
        self.slider_values[slider_idx] = slider_value # append
        if self.VERBOSE:
            print ("slider_idx:[%d] slider_value:[%.1f]"%(slider_idx,slider_value))
        
    def create_sliders(self):
        """
            Create sliders
        """
        sliders = []
        for s_idx in range(self.n_slider):
            # Create label
            if self.label_texts is None:
                label_text = "Slider %02d "%(s_idx)
            else:
                label_text = "[%d/%d]%s"%(s_idx,self.n_slider,self.label_texts[s_idx])
            slider_label = tk.Label(self.sliders_frame, text=label_text)
            slider_label.grid(row=s_idx,column=0,padx=0,pady=0)
            
            # Create slider
            if self.slider_mins is None: slider_min = 0
            else: slider_min = self.slider_mins[s_idx]
            if self.slider_maxs is None: slider_max = 100
            else: slider_max = self.slider_maxs[s_idx]
            if self.slider_vals is None: slider_val = 50
            else: slider_val = self.slider_vals[s_idx]
            slider = tk.Scale(
                self.sliders_frame,
                from_      = slider_min,
                to         = slider_max,
                orient     = tk.HORIZONTAL,
                command    = lambda value,idx=s_idx:self.cb_slider(idx,float(value)),
                resolution = self.resolution,
                length     = self.slider_width
            )
            slider.grid(row=s_idx,column=1,padx=0,pady=0,sticky=tk.W)
            slider.set(slider_val)
            sliders.append(slider)
            
        return sliders
    
    def update(self):
        if self.is_window_exists():
            self.gui.update()
        
    def run(self):
        self.gui.mainloop()
        
    def is_window_exists(self):
        try:
            return self.gui.winfo_exists()
        except tk.TclError:
            return False
        
    def get_slider_values(self):
        return self.slider_values
    
    def close(self):
        if self.is_window_exists():
            self.gui.destroy()
            self.gui.quit()
            self.gui.update()
        

### extra functions

def rpy2R(r0, order=[0,1,2]):
    c1 = np.math.cos(r0[0]); c2 = np.math.cos(r0[1]); c3 = np.math.cos(r0[2])
    s1 = np.math.sin(r0[0]); s2 = np.math.sin(r0[1]); s3 = np.math.sin(r0[2])

    a1 = np.array([[1,0,0],[0,c1,-s1],[0,s1,c1]])
    a2 = np.array([[c2,0,s2],[0,1,0],[-s2,0,c2]])
    a3 = np.array([[c3,-s3,0],[s3,c3,0],[0,0,1]])

    a_list = [a1,a2,a3]
    a = np.matmul(np.matmul(a_list[order[0]],a_list[order[1]]),a_list[order[2]])

    assert a.shape == (3,3)
    return a

def rotation(v1, v2):
    """
    Compute a matrix R that rotates v1 to align with v2.
    v1 and v2 must be length-3 1d numpy arrays.
    """
    # unit vectors
    u = v1 / np.linalg.norm(v1)
    Ru = v2 / np.linalg.norm(v2)
    # dimension of the space and identity
    dim = u.size
    I = np.identity(dim)
    # the cos angle between the vectors
    c = np.dot(u, Ru)
    # a small number
    eps = 1.0e-10
    if np.abs(c - 1.0) < eps:
        # same direction
        return I
    elif np.abs(c + 1.0) < eps:
        # opposite direction
        return -I
    else:
        # the cross product matrix of a vector to rotate around
        K = np.outer(Ru, u) - np.outer(u, Ru)
        # Rodrigues' formula
        return I + K + (K @ K) / (1 + c)

def get_uv_dict_nc(p):
    uv_dict = {}

    # Upper Body
    uv_dict['root2spine'] = np_uv(p[13,:] - p[0,:])
    uv_dict['spine2neck'] = np_uv(p[15,:] - p[13,:])
    uv_dict['neck2rs'] = np_uv(p[17,:] - p[15,:])
    uv_dict['rs2re'] = np_uv(p[18,:] - p[17,:])
    uv_dict['re2rw'] = np_uv(p[19,:] - p[18,:])
    uv_dict['neck2ls'] = np_uv(p[45,:] - p[15,:])
    uv_dict['ls2le'] = np_uv(p[46,:] - p[45,:])
    uv_dict['le2lw'] = np_uv(p[47,:] - p[46,:])

    # Lower Body
    uv_dict['root2rp'] = np_uv(p[1,:] - p[0,:])
    uv_dict['rp2rk'] = np_uv(p[2,:] - p[1,:])
    uv_dict['rk2ra'] = np_uv(p[3,:] - p[2,:])
    uv_dict['root2lp'] = np_uv(p[6,:] - p[0,:])
    uv_dict['lp2lk'] = np_uv(p[7,:] - p[6,:])
    uv_dict['lk2la'] = np_uv(p[8,:] - p[7,:])

    # Right Hand
    uv_rthumb = np_uv(np.array([0.04996, -0.04944, 0.001476]))
    cmr_rthumb = np_uv(np.array([0.0281961, -0.0225807, -0.0157689]))
    rot_rthumb = rotation(cmr_rthumb, uv_rthumb)

    uv_rindex = np_uv(np.array([0.02792, -0.09587, 0.009270]))
    cmr_rindex = np_uv(np.array([0.0227850, -0.0363963, 0.0030774]))
    rot_rindex = rotation(cmr_rindex, uv_rindex)

    uv_rmiddle = np_uv(np.array([0.008417, -0.09906, 0.01151]))
    cmr_rmiddle = np_uv(np.array([0.0043230, -0.0354591, 0.0053477]))
    rot_rmiddle = rotation(cmr_rmiddle, uv_rmiddle)

    uv_rring = np_uv(np.array([-0.01175, -0.09670, 0.01039]))
    cmr_rring = np_uv(np.array([-0.0121556, -0.0357189, 0.0033833]))
    rot_rring = rotation(cmr_rring, uv_rring)

    uv_rpinky = np_uv(np.array([-0.03086, -0.09073, 0.007162]))
    cmr_rpinky = np_uv(np.array([-0.023199, -0.0339833, -0.0007049]))
    rot_rpinky = rotation(cmr_rpinky, uv_rpinky)

    uv_dict['rw2r1meta'] = rot_rthumb.dot(np_uv(p[40,:] - p[19,:]))
    uv_dict['rw2r2meta'] = rot_rthumb.dot(np_uv(p[35,:] - p[19,:]))
    uv_dict['rw2r3meta'] = rot_rthumb.dot(np_uv(p[30,:] - p[19,:]))
    uv_dict['rw2r4meta'] = rot_rthumb.dot(np_uv(p[25,:] - p[19,:]))
    uv_dict['rw2r5meta'] = rot_rthumb.dot(np_uv(p[20,:] - p[19,:]))
    ## Thumb
    uv_dict['r1meta2r1prox'] = np_uv(p[41,:] - p[40,:])
    uv_dict['r1prox2r1dist'] = np_uv(p[42,:] - p[41,:])
    ## Index
    uv_dict['r2meta2r2prox'] = np_uv(p[36,:] - p[35,:])
    uv_dict['r2prox2r2med'] = np_uv(p[37,:] - p[36,:])
    uv_dict['r2med2r2dist'] = np_uv(p[38,:] - p[37,:])
    ## Middle
    uv_dict['r3meta2r3prox'] = np_uv(p[31,:] - p[30,:])
    uv_dict['r3prox2r3med'] = np_uv(p[32,:] - p[31,:])
    uv_dict['r3med2r3dist'] = np_uv(p[33,:] - p[32,:])
    ## Ring
    uv_dict['r4meta2r4prox'] = np_uv(p[26,:] - p[25,:])
    uv_dict['r4prox2r4med'] = np_uv(p[27,:] - p[26,:])
    uv_dict['r4med2r4dist'] = np_uv(p[28,:] - p[27,:])
    ## Pinky
    uv_dict['r5meta2r5prox'] = np_uv(p[21,:] - p[20,:])
    uv_dict['r5prox2r5med'] = np_uv(p[22,:] - p[21,:])
    uv_dict['r5med2r5dist'] = np_uv(p[23,:] - p[22,:])

    # Left Hand
    uv_lthumb = np_uv(np.array([0.04996, 0.04944, 0.001476]))
    cmr_lthumb = np_uv(np.array([0.0281961, 0.0225807, -0.0157689]))
    rot_lthumb = rotation(cmr_lthumb, uv_lthumb)

    uv_lindex = np_uv(np.array([0.02792, 0.09587, 0.009270]))
    cmr_lindex = np_uv(np.array([0.0227850, 0.0363963, 0.0030774]))
    rot_lindex = rotation(cmr_lindex, uv_lindex)

    uv_lmiddle = np_uv(np.array([0.008417, 0.09906, 0.01151]))
    cmr_lmiddle = np_uv(np.array([0.0043230, 0.0354591, 0.0053477]))
    rot_lmiddle = rotation(cmr_lmiddle, uv_lmiddle)

    uv_lring = np_uv(np.array([-0.01175, 0.09670, 0.01039]))
    cmr_lring = np_uv(np.array([-0.0121556, 0.0357189, 0.0033833]))
    rot_lring = rotation(cmr_lring, uv_lring)

    uv_lpinky = np_uv(np_uv(np.array([-0.03086, 0.09073, 0.007162])))
    cmr_lpinky = np_uv(np.array([-0.023199, 0.0339833, -0.0007049]))
    rot_lpinky = rotation(cmr_lpinky, uv_lpinky)

    uv_dict['lw2l1meta'] = rot_lthumb.dot(np_uv(p[68,:] - p[47,:]))
    uv_dict['lw2l2meta'] = rot_lthumb.dot(np_uv(p[63,:] - p[47,:]))
    uv_dict['lw2l3meta'] = rot_lthumb.dot(np_uv(p[58,:] - p[47,:]))
    uv_dict['lw2l4meta'] = rot_lthumb.dot(np_uv(p[53,:] - p[47,:]))
    uv_dict['lw2l5meta'] = rot_lthumb.dot(np_uv(p[48,:] - p[47,:]))
    ## Thumb
    uv_dict['l1meta2l1prox'] = np_uv(p[69,:] - p[68,:])
    uv_dict['l1prox2l1dist'] = np_uv(p[70,:] - p[69,:])
    ## Index
    uv_dict['l2meta2l2prox'] = np_uv(p[64,:] - p[63,:])
    uv_dict['l2prox2l2med'] = np_uv(p[65,:] - p[64,:])
    uv_dict['l2med2l2dist'] = np_uv(p[66,:] - p[65,:])
    ## Middle
    uv_dict['l3meta2l3prox'] = np_uv(p[59,:] - p[58,:])
    uv_dict['l3prox2l3med'] = np_uv(p[60,:] - p[59,:])
    uv_dict['l3med2l3dist'] = np_uv(p[61,:] - p[60,:])
    ## Ring
    uv_dict['l4meta2l4prox'] = np_uv(p[54,:] - p[53,:])
    uv_dict['l4prox2l4med'] = np_uv(p[55,:] - p[54,:])
    uv_dict['l4med2l4dist'] = np_uv(p[56,:] - p[55,:])
    ## Pinky
    uv_dict['l5meta2l5prox'] = np_uv(p[49,:] - p[48,:])
    uv_dict['l5prox2l5med'] = np_uv(p[50,:] - p[49,:])
    uv_dict['l5med2l5dist'] = np_uv(p[51,:] - p[50,:])

    return uv_dict

def get_p_target_nc(p, uv_dict):
    len_rig = {}
    len_rig['root2spine'] = 0.194939
    len_rig['spine2neck'] = 0.2263381
    len_rig['neck2rs'] = 0.1832940
    len_rig['rs2re'] = 0.291422
    len_rig['re2rw'] = 0.270311
    len_rig['neck2ls'] = 0.1832940
    len_rig['ls2le'] = 0.291422
    len_rig['le2lw'] = 0.270311
    len_rig['root2rp'] = 0.100371
    len_rig['rp2rk'] = 0.4454116
    len_rig['rk2ra'] = 0.436307
    len_rig['root2lp'] = 0.100371
    len_rig['lp2lk'] = 0.4454116
    len_rig['lk2la'] = 0.436307

    p_target = {}
    p_target['right_pelvis'] = p[0,:] + len_rig['root2rp'] * uv_dict['root2rp']
    p_target['right_knee'] = p_target['right_pelvis'] + len_rig['rp2rk'] * uv_dict['rp2rk']
    p_target['right_ankle'] = p_target['right_knee'] + len_rig['rk2ra'] * uv_dict['rk2ra']
    p_target['left_pelvis'] = p[0,:] + len_rig['root2lp'] * uv_dict['root2lp']
    p_target['left_knee'] = p_target['left_pelvis'] + len_rig['lp2lk'] * uv_dict['lp2lk']
    p_target['left_ankle'] = p_target['left_knee'] + len_rig['lk2la'] * uv_dict['lk2la']
    p_target['spine'] = p[0,:] + len_rig['root2spine'] * uv_dict['root2spine']
    p_target['neck'] = p_target['spine'] + len_rig['spine2neck'] * uv_dict['spine2neck']
    p_target['right_shoulder'] = p_target['neck'] + len_rig['neck2rs'] * uv_dict['neck2rs']
    p_target['right_elbow'] = p_target['right_shoulder'] + len_rig['rs2re'] * uv_dict['rs2re']
    p_target['right_wrist'] = p_target['right_elbow'] + len_rig['re2rw'] * uv_dict['re2rw']
    p_target['left_shoulder'] = p_target['neck'] + len_rig['neck2ls'] * uv_dict['neck2ls']
    p_target['left_elbow'] = p_target['left_shoulder'] + len_rig['ls2le'] * uv_dict['ls2le']
    p_target['left_wrist'] = p_target['left_elbow'] + len_rig['le2lw'] * uv_dict['le2lw']

    len_rig['rw2r1meta'] = 0.07030502
    len_rig['rw2r2meta'] = 0.10028379
    len_rig['rw2r3meta'] = 0.100085855
    len_rig['rw2r4meta'] = 0.09796863
    len_rig['rw2r5meta'] = 0.09610762

    len_rig['r1meta2r1prox'] = 0.03398629
    len_rig['r1prox2r1dist'] = 0.02548997

    len_rig['r2meta2r2prox'] = 0.03861091
    len_rig['r2prox2r2med'] = 0.033236664
    len_rig['r2med2r2dist'] = 0.024927637

    len_rig['r3meta2r3prox'] = 0.03860989
    len_rig['r3prox2r3med'] = 0.03323699
    len_rig['r3med2r3dist'] = 0.024926964

    len_rig['r4meta2r4prox'] = 0.0386101
    len_rig['r4prox2r4med'] = 0.033236515
    len_rig['r4med2r4dist'] = 0.02492442

    len_rig['r5meta2r5prox'] = 0.038610853
    len_rig['r5prox2r5med'] = 0.03323685
    len_rig['r5med2r5dist'] = 0.02492764

    len_rig['lw2l1meta'] = 0.07030502
    len_rig['lw2l2meta'] = 0.10028379
    len_rig['lw2l3meta'] = 0.100085855
    len_rig['lw2l4meta'] = 0.09796863
    len_rig['lw2l5meta'] = 0.09610762

    len_rig['l1meta2l1prox'] = 0.03398629
    len_rig['l1prox2l1dist'] = 0.02548997

    len_rig['l2meta2l2prox'] = 0.03861091
    len_rig['l2prox2l2med'] = 0.033236664
    len_rig['l2med2l2dist'] = 0.024927637

    len_rig['l3meta2l3prox'] = 0.03860989
    len_rig['l3prox2l3med'] = 0.03323699
    len_rig['l3med2l3dist'] = 0.024926964

    len_rig['l4meta2l4prox'] = 0.0386101
    len_rig['l4prox2l4med'] = 0.033236515
    len_rig['l4med2l4dist'] = 0.02492442

    len_rig['l5meta2l5prox'] = 0.03861085
    len_rig['l5prox2l5med'] = 0.03323685
    len_rig['l5med2l5dist'] = 0.02492764

    p_target['rthumb_l2'] = len_rig['rw2r1meta'] * uv_dict['rw2r1meta']
    p_target['rindex_l1'] = len_rig['rw2r2meta'] * uv_dict['rw2r2meta']
    p_target['rmiddle_l1'] = len_rig['rw2r3meta'] * uv_dict['rw2r3meta']
    p_target['rring_l1'] = len_rig['rw2r4meta'] * uv_dict['rw2r4meta']
    p_target['rpinky_l1'] = len_rig['rw2r5meta'] * uv_dict['rw2r5meta']
    
    p_target['rthumb_l3'] = p_target['rthumb_l2'] + len_rig['r1meta2r1prox'] * uv_dict['r1meta2r1prox']
    p_target['rthumb_end'] = p_target['rthumb_l3'] + len_rig['r1prox2r1dist'] * uv_dict['r1prox2r1dist']

    p_target['rindex_l2'] = p_target['rindex_l1'] + len_rig['r2meta2r2prox'] * uv_dict['r2meta2r2prox']
    p_target['rindex_l3'] = p_target['rindex_l2'] + len_rig['r2prox2r2med'] * uv_dict['r2prox2r2med']
    p_target['rindex_end'] = p_target['rindex_l3'] + len_rig['r2med2r2dist'] * uv_dict['r2med2r2dist']

    p_target['rmiddle_l2'] = p_target['rmiddle_l1'] + len_rig['r3meta2r3prox'] * uv_dict['r3meta2r3prox']
    p_target['rmiddle_l3'] = p_target['rmiddle_l2'] + len_rig['r3prox2r3med'] * uv_dict['r3prox2r3med']
    p_target['rmiddle_end'] = p_target['rmiddle_l3'] + len_rig['r3med2r3dist'] * uv_dict['r3med2r3dist']

    p_target['rring_l2'] = p_target['rring_l1'] + len_rig['r4meta2r4prox'] * uv_dict['r4meta2r4prox']
    p_target['rring_l3'] = p_target['rring_l2'] + len_rig['r4prox2r4med'] * uv_dict['r4prox2r4med']
    p_target['rring_end'] = p_target['rring_l3'] + len_rig['r4med2r4dist'] * uv_dict['r4med2r4dist']

    p_target['rpinky_l2'] = p_target['rpinky_l1'] + len_rig['r5meta2r5prox'] * uv_dict['r5meta2r5prox']
    p_target['rpinky_l3'] = p_target['rpinky_l2'] + len_rig['r5prox2r5med'] * uv_dict['r5prox2r5med']
    p_target['rpinky_end'] = p_target['rpinky_l3'] + len_rig['r5med2r5dist'] * uv_dict['r5med2r5dist']

    p_target['lthumb_l2'] = len_rig['lw2l1meta'] * uv_dict['lw2l1meta']
    p_target['lindex_l1'] = len_rig['lw2l2meta'] * uv_dict['lw2l2meta']
    p_target['lmiddle_l1'] = len_rig['lw2l3meta'] * uv_dict['lw2l3meta']
    p_target['lring_l1'] = len_rig['lw2l4meta'] * uv_dict['lw2l4meta']
    p_target['lpinky_l1'] = len_rig['lw2l5meta'] * uv_dict['lw2l5meta']

    p_target['lthumb_l3'] = p_target['lthumb_l2'] + len_rig['l1meta2l1prox'] * uv_dict['l1meta2l1prox']
    p_target['lthumb_end'] = p_target['lthumb_l3'] + len_rig['l1prox2l1dist'] * uv_dict['l1prox2l1dist']

    p_target['lindex_l2'] = p_target['lindex_l1'] + len_rig['l2meta2l2prox'] * uv_dict['l2meta2l2prox']
    p_target['lindex_l3'] = p_target['lindex_l2'] + len_rig['l2prox2l2med'] * uv_dict['l2prox2l2med']
    p_target['lindex_end'] = p_target['lindex_l3'] + len_rig['l2med2l2dist'] * uv_dict['l2med2l2dist']

    p_target['lmiddle_l2'] = p_target['lmiddle_l1'] + len_rig['l3meta2l3prox'] * uv_dict['l3meta2l3prox']
    p_target['lmiddle_l3'] = p_target['lmiddle_l2'] + len_rig['l3prox2l3med'] * uv_dict['l3prox2l3med']
    p_target['lmiddle_end'] = p_target['lmiddle_l3'] + len_rig['l3med2l3dist'] * uv_dict['l3med2l3dist']

    p_target['lring_l2'] = p_target['lring_l1'] + len_rig['l4meta2l4prox'] * uv_dict['l4meta2l4prox']
    p_target['lring_l3'] = p_target['lring_l2'] + len_rig['l4prox2l4med'] * uv_dict['l4prox2l4med']
    p_target['lring_end'] = p_target['lring_l3'] + len_rig['l4med2l4dist'] * uv_dict['l4med2l4dist']

    p_target['lpinky_l2'] = p_target['lpinky_l1'] + len_rig['l5meta2l5prox'] * uv_dict['l5meta2l5prox']
    p_target['lpinky_l3'] = p_target['lpinky_l2'] + len_rig['l5prox2l5med'] * uv_dict['l5prox2l5med']
    p_target['lpinky_end'] = p_target['lpinky_l3'] + len_rig['l5med2l5dist'] * uv_dict['l5med2l5dist']

    return p_target