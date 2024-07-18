import numpy as np

def main(args):
    from env.ur5_env import UR5Env
    
    env = UR5Env(args)

    torq = np.zeros((6,))
    
    while env.is_alive:
        env.step(torq)

    env.stop()

if __name__ == "__main__":

    args = {}
    args['is_render']  = True
    args['xml_file']   = 'ur5e.xml'
    args['cam_azi'] = 90
    args['cam_ele'] = -20
    args['cam_dist'] =  5

    main(args)