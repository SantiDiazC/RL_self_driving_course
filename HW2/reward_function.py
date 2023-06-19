def reward_function(params):
    '''
    Example of rewarding the agent to stay inside two borders
    and penalizing getting too close to the objects in front
    '''
    import math
    all_wheels_on_track = params['all_wheels_on_track']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    objects_distance = params['objects_distance']
    _, next_object_index = params['closest_objects']
    objects_left_of_center = params['objects_left_of_center']
    is_left_of_center = params['is_left_of_center']
    steps = params['steps']
    progress = params['progress']
    crashed = params['if_crashed']

    TOTAL_NUM_STEPS = 300

    # Read input variables
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']

    # Calculate the direction of the center line based on the closest waypoints
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]

    # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
    track_direction = math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])

    # Convert to degree
    track_direction = math.degrees(track_direction)

    # Calculate the difference between the track direction and the heading direction of the car
    direction_diff = abs(track_direction - heading)

    # Penalize the reward if the difference is too large
    DIRECTION_THRESHOLD = 10.0

    malus = 1
    reward_dir = 1.0
    if direction_diff > DIRECTION_THRESHOLD:
        malus = 1 - (direction_diff / 50)
        if malus < 0 or malus > 1:
            malus = 0
        reward_dir *= malus


    # Initialize reward with a small number but not zero
    # because zero means off-track or crashed
    reward = 1e-3

    # Reward if the agent stays inside the two borders of the track
    if all_wheels_on_track and (0.5 * track_width - distance_from_center) >= 0.05:
        reward_lane = 1.0
    else:
        reward_lane = -3.0

    # Penalize if the agent is too close to the next object
    reward_avoid = 1.0

    reward_progress = 0.1*progress
    # Distance to the next object
    distance_closest_object = objects_distance[next_object_index]
    # Decide if the agent and the next object is on the same lane
    is_same_lane = objects_left_of_center[next_object_index] == is_left_of_center

    if is_same_lane:
        if 0.5 <= distance_closest_object < 0.8:
            reward_avoid *= 0.5
        elif 0.3 <= distance_closest_object < 0.5:
            reward_avoid *= 0.3
        elif distance_closest_object < 0.3:
            reward_avoid = -10.0 # Likely crashed
        # Give additional reward if the car pass every 100 steps faster than expected
    if crashed:
        reward_avoid += -50.0


    if (steps % 100) == 0 and progress > (steps / TOTAL_NUM_STEPS) * 100:
        reward_progress += 20.0
    # Calculate reward by putting different weights on
    # the two aspects above
    reward += 1.0 * reward_lane + 4.0 * reward_avoid + 1.0*reward_progress
    reward *= reward_dir
    return reward
