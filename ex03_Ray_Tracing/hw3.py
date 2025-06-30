from helper_classes import *
import matplotlib.pyplot as plt

class Scene:
    def __init__(self, camera, ambient, lights, objects, screen_size):
        self.camera = camera
        self.ambient = ambient
        self.lights = lights
        self.objects = objects
        self.screen_size = screen_size

def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)
            color = np.zeros(3)

            # This is the main loop where each pixel color is computed.
            scene = Scene(camera, ambient, lights, objects, screen_size)
            intersection = find_intersection(ray, objects)
            if intersection is not None:
                hit_point, obj = intersection
                hit_point += get_point_bias(obj, hit_point)
                color = get_color(
                    scene, obj, ray, hit_point, 0, max_depth)

            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)

    return image


def find_intersection(ray, objects):
    # Ask the ray which object it intersects first
    t, object = ray.nearest_intersected_object(objects)
    # No hit = early out
    if object is None:
        return None
    # Convert parametric distance 't' to an actual Cartesian point:
    hit_point = ray.origin + ray.direction * t
    return hit_point, object


def get_color(scene, closest_object, ray, intersection_point, depth, max_depth):
    color = np.zeros(3)
    
    # Ambient component
    color += calc_ambient_color(scene.ambient, closest_object)

    # Diffuse and specular lighting
    for light in scene.lights:
        shadow_coefficient = calc_shadow_coefficient(
            light, intersection_point, scene.objects)
        if shadow_coefficient > 0:
            color += calc_diffuse_color(intersection_point, light, closest_object)
            color += calc_specular_color(scene, intersection_point, light, closest_object)

    # Recursive effects
    depth += 1
    if depth < max_depth:
        # Reflection
        if closest_object.reflection > 0:
            reflected_ray = construct_reflective_ray(
                intersection_point, ray, closest_object)
            intersection = find_intersection(reflected_ray, scene.objects)
            if intersection is not None:
                hit_point, obj = intersection
                hit_point += get_point_bias(obj, hit_point)
                reflected_color = get_color(
                    scene, obj, reflected_ray, hit_point, depth, max_depth)
                color += closest_object.reflection * reflected_color

        # Refraction (lazy implementation - same direction)
        if closest_object.refraction > 0:
            # Small bias in ray direction to avoid self-intersection
            bias = 1e-5 * ray.direction
            refracted_ray = Ray(intersection_point + bias, ray.direction)
            
            # Find next intersection (ignore current object to prevent self-hit)
            remaining_objects = [obj for obj in scene.objects if obj != closest_object]
            intersection = find_intersection(refracted_ray, remaining_objects)
            
            if intersection is not None:
                hit_point, obj = intersection
                hit_point += get_point_bias(obj, hit_point)
                refracted_color = get_color(
                    scene, obj, refracted_ray, hit_point, depth, max_depth)
                color += closest_object.refraction * refracted_color
            else:
                # Use background color if nothing is hit
                color += closest_object.refraction * scene.ambient

    return np.clip(color, 0, 1)  # Ensure color stays in valid range


def calc_ambient_color(ambient, object):
    return ambient * object.ambient


def calc_diffuse_color(hit_point, light, object):
    N = calc_object_norm(object, hit_point)
    L_direction = light.get_light_ray(hit_point).direction

    return light.get_intensity(hit_point) * object.diffuse * np.dot(N, L_direction)


def calc_specular_color(scene, hit_point, light, object):
    V_hat = normalize(scene.camera - hit_point)
    L_direction = -1 * light.get_light_ray(hit_point).direction
    R_hat = reflected(L_direction, calc_object_norm(object, hit_point))

    return light.get_intensity(hit_point) * object.specular * (np.dot(V_hat, R_hat) ** object.shininess)


def construct_reflective_ray(hit_point, ray, object):
    return Ray(hit_point, reflected(ray.direction, calc_object_norm(object, hit_point)))


def calc_shadow_coefficient(light, hit_point, objects):
    ray = light.get_light_ray(hit_point)
    distance, object = ray.nearest_intersected_object(objects)

    if object is None:
        return 1

    if distance < light.get_distance_from_light(hit_point):
        return 0

    return 1


def get_point_bias(object, hit_point):
    return 0.01 * object.compute_normal(hit_point)


def calc_object_norm(object, hit_point):
    return object.compute_normal(hit_point)


def your_own_scene():
    """
    Objects:
        • 1 blue glass-like sphere
        • 1 red diamond-shaped pyramid
        • 1 grey ground plane
    """
    sphere = Sphere(center=[0, -0.01, -1], radius=0.7)
    sphere.set_material([0.1, 0.1, 0.5], [0.1, 0.1, 0.5],
                        [0.5, 0.5, 0.5], 32, 0.5, 0.9)

    pyramid = Diamond(v_list=[
        [0.5, -0.5, -1.5],
        [1.5, -0.5, -1.5],
        [1.0, -0.5, -2.5],
        [1.0, 0.5, -1.5],
        [1.0, -1.5, -1.5]
    ])
    pyramid.set_material([0.5, 0.1, 0.1], [0.5, 0.1, 0.1], [
                         0.5, 0.5, 0.5], 32, 0.3)
    pyramid.apply_materials_to_triangles()

    plane = Plane(normal=[0, 1, 0], point=[0, -0.5, 0])
    plane.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2],
                       [0.5, 0.5, 0.5], 32, 0.5)

    objects = [sphere, pyramid, plane]

    point_light = PointLight(intensity=np.array(
        [1, 1, 1]), position=np.array([1, 1, 1]), kc=0.1, kl=0.1, kq=0.1)
    directional_light = DirectionalLight(intensity=np.array(
        [1, 1, 1]), direction=np.array([-1, -1, -1]))
    lights = [point_light, directional_light]

    camera = np.array([0, 0, 1])

    return camera, lights, objects


def tiny_sailboat_scene():
    """
    Objects:
        • 2 triangles for the wooden hull
        • 1 triangle for the mainsail
        • 1 triangle for the front sail (Jib)
        • 1 ground plane
    """

    # Put the boat 3 units in front of the camera, centred on X-axis
    hull_back   = np.array([-0.9, -0.7, -3.0])
    hull_front  = np.array([ 0.9, -0.7, -3.0])
    hull_tip    = np.array([ 0.0, -0.3, -3.7])   # pointy bow

    deck_back   = np.array([-0.7, -0.5, -3.0])
    deck_front  = np.array([ 0.7, -0.5, -3.0])
    deck_tip    = np.array([ 0.0, -0.1, -3.7])   # upper edge of hull

    mast_base   = np.array([ 0.0, -0.1, -3.2])
    mast_top    = np.array([ 0.0,  0.9, -3.4])

    jib_lead    = np.array([ 0.55, -0.1, -3.35]) # small front sail corner


    # Hull
    hull_lower = Triangle(hull_back, hull_front, hull_tip)
    hull_upper = Triangle(deck_back, deck_tip, deck_front)

    wood = [0.55, 0.27, 0.07]
    for tri in (hull_lower, hull_upper):
        tri.set_material(
            ambient = wood,
            diffuse = wood,
            specular = [0.3, 0.3, 0.3],
            shininess = 12,
            reflection = 0.15
        )

    # Mainsail (white)
    mainsail = Triangle(mast_base, mast_top, deck_tip)
    mainsail.set_material(
        ambient = [1.0, 1.0, 1.0],
        diffuse = [1.0, 1.0, 1.0],
        specular = [0.4, 0.4, 0.4],
        shininess = 25,
        reflection = 0.05
    )

    # Jib
    jib = Triangle(mast_base, jib_lead, mast_top)
    jib.set_material(
        ambient = [0.9, 0.9, 0.9],
        diffuse = [0.9, 0.9, 0.9],
        specular = [0.4, 0.4, 0.4],
        shininess = 20,
        reflection = 0.05
    )

    # Ground plane
    sea = Plane(normal=[0, 1, 0], point=[0, -1.5, 0])
    sea.set_material(
        ambient   = [0.05, 0.05, 0.12],
        diffuse   = [0.05, 0.05, 0.25],
        specular  = [0.6, 0.6, 0.8],
        shininess = 50,
        reflection = 0.4
    )

    lights = [
        # Warm sun
        PointLight(intensity=np.array([1.2, 1.1, 0.9]),
                   position=np.array([ 2, 3,  1]),
                   kc=1.0, kl=0.09, kq=0.032),
    ]

    camera = np.array([0, 0, 1])

    objects = [hull_lower, hull_upper, mainsail, jib, sea]
    return camera, lights, objects