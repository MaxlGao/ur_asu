import numpy as np

# Virtual table level in mm
VERTICAL_OFFSET = 0.003 # 0.000 = gripper tips always on table (dangerous)

GRIPPER_TABLE = { # Known, measured values. Gripper width in 0.1mm.
       0: 0.153,
     200: 0.150,
     400: 0.147,
     600: 0.141,
     800: 0.132,
    1000: 0.114,
}

def gripper_width_to_height(gripper_width):
    # Clamp input to valid range
    if gripper_width <= 0:
        return GRIPPER_TABLE[0]
    if gripper_width >= 1000:
        return GRIPPER_TABLE[1000]

    # Sort the keys to find where gripper_width fits
    keys = sorted(GRIPPER_TABLE.keys())
    for i in range(len(keys) - 1):
        low = keys[i]
        high = keys[i + 1]
        if low <= gripper_width <= high:
            # Linear interpolation
            low_val = GRIPPER_TABLE[low]
            high_val = GRIPPER_TABLE[high]
            t = (gripper_width - low) / (high - low)
            return low_val + t * (high_val - low_val)

def height_to_gripper_width(height):
    # Convert GRIPPER_TABLE to (height -> width) tuples for inverse lookup
    items = sorted(((v, k) for k, v in GRIPPER_TABLE.items()), reverse=True)

    # Clamp to valid range
    if height >= items[0][0]:
        return items[0][1]
    if height <= items[-1][0]:
        return items[-1][1]

    # Find surrounding interval
    for i in range(len(items) - 1):
        high_val, low_key = items[i]
        low_val, high_key = items[i + 1]
        if high_val >= height >= low_val:
            t = (height - low_val) / (high_val - low_val)
            return int(high_key + t * (low_key - high_key))

def EE_pose_to_pushers_2D(EE_pose):
    """
    Returns the gripper positions given the current EE pose, assuming the gripper width is set according to height.
    Currently doesn't intelligently distinguish which gripper's which.
    Pusher 1 is the left one when looking down the camera.
    Pusher positions are [x, y, 0]
    """
    position, orientation = EE_pose
    center_x, center_y, height = position
    height -= VERTICAL_OFFSET
    yaw = orientation[2]
    yaw_r = np.deg2rad(yaw)
    gripper_width = height_to_gripper_width(height) * 0.0001 # Conversion to meters
    half_width = gripper_width / 2

    dx = half_width * np.cos(yaw_r)
    dy = half_width * np.sin(yaw_r)
    pusher1_position = [center_x + dx, center_y + dy, 0.0]
    pusher2_position = [center_x - dx, center_y - dy, 0.0]

    return pusher1_position, pusher2_position

def pushers_to_EE_pose_2D(pusher_1_position, pusher_2_position, given_yaw=0):
    """
    Returns the EE pose such that the gripper can place its two tips at the given XY locations
    pusher_positions are [x1, y1, z1], [x2, y2, z2] meters, where zs are ignored
    returns position, orientation_deg
    Prone to singularity (pushers at the same point)
    """
    pos1_np = np.array(pusher_1_position)
    pos2_np = np.array(pusher_2_position)
    gripper_width = np.linalg.norm(pos1_np-pos2_np)
    # Assert that the two points are within the gripper maximum (110 mm in this case)
    if gripper_width > 0.11:
        return None
    gripper_height = gripper_width_to_height(gripper_width*10000) # conversion to their weird measurement
    center = (pos1_np + pos2_np) / 2
    position = [center[0], center[1], gripper_height + VERTICAL_OFFSET]
    if gripper_width < 0.01:
        yaw = given_yaw
    else:
        # If gripper width is more than 10 mm, probably okay to measure angle.
        yaw = np.atan2(pos1_np[1]-pos2_np[1], pos1_np[0]-pos2_np[0])
        yaw = np.rad2deg(yaw)
        print(f"YAW IS {yaw}")
    orientation = [0, 180, yaw]
    return position, orientation

def pointspan_to_pushers(pusher_1, span, yaw_d):
    """Provides a pair of pusher positions given pusher 1's position, span, and the angle"""
    yaw_r = np.deg2rad(yaw_d)
    dx = span * np.cos(yaw_r)
    dy = span * np.sin(yaw_r)
    new_x = pusher_1[0] - dx
    new_y = pusher_1[1] - dy
    pusher_2 = [new_x, new_y, 0.0]
    return pusher_1, pusher_2

def pointspan_to_EE_pose_2D(pusher_1, span, yaw_d):
    """
    Provides a EE pose given pusher 1's position, span, and the angle
    Pusher positions are [x, y, --] meters
    given yaw is in degrees
    span is in meters
    pose is [position, euler orientation degrees]
    """
    yaw_r = np.deg2rad(yaw_d)
    half_span = span/2
    dx = half_span * np.cos(yaw_r)
    dy = half_span * np.sin(yaw_r)
    cen_x = pusher_1[0] - dx
    cen_y = pusher_1[1] - dy
    gripper_height = gripper_width_to_height(span*10000) # conversion to their weird measurement
    position = [cen_x, cen_y, gripper_height + VERTICAL_OFFSET]
    orientation = [0, 180, yaw_d]
    return position, orientation

def EE_pose_to_pointspan(EE_pose):
    """
    Returns the pointspan given the current EE pose, assuming the gripper width is set according to height.
    Pusher 1 is the left one when looking down the camera.
    Pusher positions are [x, y, 0]
    """
    position, orientation = EE_pose
    center_x, center_y, height = position
    height -= VERTICAL_OFFSET
    yaw_d = orientation[2]
    yaw_r = np.deg2rad(yaw_d)
    gripper_width = height_to_gripper_width(height) * 0.0001 # Conversion to meters
    half_width = gripper_width / 2

    dx = half_width * np.cos(yaw_r)
    dy = half_width * np.sin(yaw_r)
    pusher1_position = [center_x + dx, center_y + dy, 0.0]

    return pusher1_position, gripper_width, yaw_d

# def pointspan_to_trajectories(EE_pose_now, pusher_1_target, span_target, yaw_target, duration):
#     """
#     Provides a list of joint angle trajectories based on where you want your pushers
#     Pusher positions are [x, y, --] meters
#     given yaw is in degrees
#     span is in meters
#     pose is [position, euler orientation degrees]
#     """
#     print(f"""We're currently at:
# {pose_text(EE_pose_now)}""")
#     pusher_1_now, pusher_2_now = EE_pose_to_pushers_2D(EE_pose_now)
#     print("pushers now")
#     print(pushers_text(pusher_1_now, pusher_2_now))
#     pos1_now_np = np.array(pusher_1_now)
#     pos2_now_np = np.array(pusher_2_now)
#     pusher_1_target, pusher_2_target = pointspan_to_pushers(pusher_1_target, span_target, yaw_target)
#     print("pushers target")
#     print(pushers_text(pusher_1_target, pusher_2_target))
#     pos1_target_np = np.array(pusher_1_target)
#     pos2_target_np = np.array(pusher_2_target)
#     width_now = np.linalg.norm(pos1_now_np - pos2_now_np)
#     width_target = np.linalg.norm(pos1_target_np - pos2_target_np)
#     # Checks if the 40mm width mark is crossed
#     if width_now <= 0.04 <= width_target or width_target <= 0.04 <= width_now:
#         # If it is, we need an intermediate waypoint. 
#         # Otherwise, the controller will increase/decrease height linearly, which can bump the gripper into the table.
#         # This is to have the controller raise/lower more slowly when it matters most.

#         # If pushers move linearly at constant speed, then width will increase linearly
#         # We need to find at what point does width = 0.04.
#         proportion = abs(0.04 - width_now) / abs(width_target - width_now)
#         # Then, find the vector each pusher travels along and find the appropriate point.
#         vec1 = pos1_target_np - pos1_now_np
#         vec2 = pos2_target_np - pos2_now_np
#         prop_vec1 = proportion * vec1
#         prop_vec2 = proportion * vec2
#         intermed_pos1 = pos1_now_np + prop_vec1
#         intermed_pos2 = pos2_now_np + prop_vec2
#         print("pushers intermed")
#         print(pushers_text(intermed_pos1, intermed_pos2))
#         intermed_position, intermed_rpy = pushers_to_EE_pose_2D(intermed_pos1, intermed_pos2)
#         position, rpy = pointspan_to_EE_pose_2D(pusher_1_target, span_target, yaw_target)
#         print(f"""Got coordinates: 
# {pose_text([intermed_position, intermed_rpy])}
#            then:
# {pose_text([position, rpy])}""")
#         return [move(intermed_position, intermed_rpy, duration), move(position, rpy, duration)]
#     else:
#         position, rpy = pointspan_to_EE_pose_2D(pusher_1_target, span_target, yaw_target)
#         print(f"""Got coordinates: 
# {pose_text([position, rpy])}""")
#         return [move(position, rpy, duration)]