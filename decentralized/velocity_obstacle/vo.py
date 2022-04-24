'''
Velocity Obstacles
'''

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import numpy as np
from zmq import BINDTODEVICE

SIM_TIME            = 5.0   # Total simulation time
TIMESTEP            = 0.1   # Time step for each simulation
ROBOT_RADIUS        = 0.5   # Radius of the robot
VMAX                = 2     # Maximum Velocity of the Robot
VMIN                = 0.2   # Minimum Velocity of the Robot    
NUMBER_OF_TIMESTEPS = int(SIM_TIME/TIMESTEP)    # Number of timesteps in a given simulation

def simulate():
    # Creating obstacles
    obstacles = create_obstacles(SIM_TIME,NUMBER_OF_TIMESTEPS)

    start = np.array([5,0,0,0]) # Start position of the robot
    goal = np.array([5,10,0,0]) # Goal position of the robot

    # Initiating the start position
    robot_state = start

    # Initializing the robot history list
    robot_state_hist = np.empty((4,NUMBER_OF_TIMESTEPS))
    
    
    for i in range(NUMBER_OF_TIMESTEPS):
        # Calculating the desired veloity for the robot
        v_desired   = compute_desired_velocity(robot_state, goal, ROBOT_RADIUS, VMAX)
        control_vel = compute_velocity(robot_state, obstacles[:, i, :], v_desired)
        robot_state = update_state(robot_state, control_vel)
        robot_state_history[:4, i] = robot_state



#**********************************
''' COMPUTING VELOCITY OBSTACLE '''
#**********************************
def compute_velocity(robot, obstacles, v_desired):
    '''
    Inputs:
        robot       - Robot current state 
        obstacles   - all the obstacles position and velcity computed for that time instant
        v_Desired   - Desired velocity for the robot computed at that time instant
    Output:
        cmd_vel     - Velocity of the robot with acceleration constraints
    '''
    pA = robot[:2]      # Current position of the robot
    vA = robot[-2:]     # Current velocity of the robot

    number_of_obstacles = np.shape(obstacles)[1] # Finding the total number of obstacles present
    
    Amat                = np.empty((number_of_obstacles*2,2))
    bvec                = np.empty((number_of_obstacles*2))

    for i in range(number_of_obstacles):
        obstacle = obstacles[:,i]



#**********************************************
''' COMPUTING DESIRED VELOCITY OF THE ROBOT '''
#**********************************************
def compute_desired_velocity(current_pos, goal_pos, robot_radius, vmax):
    '''
    Input:
        current_pos     - Current position of the robot
        goal_pos        - The end or goal position where are robot should reach
        robot_radius    - Radius of the mobile robot
        vmax            - Maximum velocity the mobile robot can travel
    Output:
        desired_Vel     - Desired velocity for the mobile robot
    '''
    disp_vec = (goal_pos - current_pos)[:2] # The displacement vector from current position to goal position is calcualted i.e. (del_x, del_y)
    norm = np.linalg.norm(disp_vec)         # The magnitude of the displacement vector is calculated
    if norm < robot_radius / 5:             # Checking if the displacement magnitutde is less than 1/5th of the robot radius
        return np.zeros(2)                  # If so the desired velocity is made zero, assuming the robot has reached goal
    
    disp_vec = disp_vec / norm              # Calculating the unit-vector for direction of goal
    # np.shape(disp_vec)                      
    desired_vel = vmax * disp_vec           # Calculating the desired velocity in the x and y direction based on unit vector

    return desired_vel



##############################
''' CREATING OBSTACLES '''
##############################
def create_obstacles(sim_time, num_timesteps):
    # Obstacle 1
    v = -2
    p0 = np.array([5, 12])
    obst = create_robot(p0, v, np.pi/2, sim_time,num_timesteps).reshape(4, num_timesteps, 1)
    obstacles = obst
    # Obstacle 2
    v = 2
    p0 = np.array([0, 5])
    obst = create_robot(p0, v, 0, sim_time, num_timesteps).reshape(4, num_timesteps, 1)
    obstacles = np.dstack((obstacles, obst))
    # Obstacle 3
    v = 2
    p0 = np.array([10, 10])
    obst = create_robot(p0, v, -np.pi * 3 / 4, sim_time, num_timesteps).reshape(4,num_timesteps, 1)
    obstacles = np.dstack((obstacles, obst))
    # Obstacle 4
    v = 2
    p0 = np.array([7.5, 2.5])
    obst = create_robot(p0, v, np.pi * 3 / 4, sim_time, num_timesteps).reshape(4,num_timesteps, 1)
    obstacles = np.dstack((obstacles, obst))
    
    # # Obstacle 1
    # v = 0
    # p0 = np.array([2, 2])
    # obst = create_robot(p0, v, np.pi/2, sim_time,num_timesteps).reshape(4, num_timesteps, 1)
    # obstacles = obst

    # # Obstacle 2
    # v = 0
    # p0 = np.array([8, 8])
    # obst = create_robot(p0, v, np.pi * 3 / 4, sim_time, num_timesteps).reshape(4,num_timesteps, 1)
    # obstacles = np.dstack((obstacles, obst))  
    
    return obstacles


def create_robot(p0, v, theta, sim_time, num_timesteps):
    # Creates obstacles starting at p0 and moving at v in theta direction
    t = np.linspace(0, sim_time, num_timesteps)
    theta = theta * np.ones(np.shape(t))
    vx = v * np.cos(theta)
    vy = v * np.sin(theta)
    v = np.stack([vx, vy])
    p0 = p0.reshape((2, 1))
    p = p0 + np.cumsum(v, axis=1) * (sim_time / num_timesteps)
    p = np.concatenate((p, v))
    return p



