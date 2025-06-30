import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis


## Lights

class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = normalize(direction)

    # This function returns the ray that goes from the light source to a point

    def get_light_ray(self, intersection_point):
        return Ray(intersection_point, -self.direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.inf

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        # calculate distance between light source and intersection 
        # calculate and return the light intensity based on kc, kl, kq
        d = self.get_distance_from_light(intersection)
        f_att = 1 / (self.kc + self.kl*d + self.kq * (d**2))
        return self.intensity * f_att


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.direction = normalize(direction)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        f_att = 1 / (self.kc + self.kl*d + self.kq * (d**2))
        V = normalize(self.get_light_ray(intersection).direction)

        return self.intensity * f_att * np.dot(V, -self.direction)


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    
    def nearest_intersected_object(self, objects):
        nearest_object = None
        min_distance = np.inf
        for object in objects:
            intersection = object.intersect(self)
            if intersection is not None:
                t, _ = intersection
                if t < min_distance:
                    min_distance = t
                    nearest_object = object
        return min_distance, nearest_object


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection, refraction=0.0):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection
        self.refraction = refraction


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def compute_normal(self, _):
        return self.normal

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = np.dot(v, self.normal) / (np.dot(self.normal, ray.direction) + 1e-6)
        if t > 0:
            return t, self
        else:
            return None


class Triangle(Object3D):
    """
        C
        /\
       /  \
    A /____\ B

    The fornt face of the triangle is A -> B -> C.
    
    """
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.compute_normal(a)

    # computes normal to the trainagle surface. Pay attention to its direction!
    def compute_normal(self, _):
        return normalize(np.cross(self.b - self.a, self.c - self.a))

    def intersect(self, ray: Ray):
        triangle_plane = Plane(self.normal, self.a)
        intersection = triangle_plane.intersect(ray)
        if intersection is None:
            return None

        t, _ = intersection
        intersection_point = ray.origin + t * ray.direction
        if self.barrycentric(intersection_point):
            return intersection
        return None
    
    def barrycentric(self, intersection_point):
        p = intersection_point
        pa = p - self.a
        pb = p - self.b
        pc = p - self.c

        area_ABC = np.linalg.norm(np.cross(self.b - self.a, self.c - self.a))
        area_PBC = np.linalg.norm(np.cross(pb, pc))
        area_PCA = np.linalg.norm(np.cross(pc, pa))
        area_PAB = np.linalg.norm(np.cross(pa, pb))

        alpha = area_PBC / area_ABC
        beta = area_PCA / area_ABC
        gamma = area_PAB / area_ABC

        return self.check_intersect_condition(alpha, beta, gamma)

    def check_intersect_condition(self, alpha, beta, gamma):
        return 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1 and np.abs(alpha+beta+gamma - 1) < 1e-6
    


class Diamond(Object3D):
    """     
            D
            /\*\
           /==\**\
         /======\***\
       /==========\***\
     /==============\****\
   /==================\*****\
A /&&&&&&&&&&&&&&&&&&&&\ B &&&/ C
   \==================/****/
     \==============/****/
       \==========/****/
         \======/***/
           \==/**/
            \/*/
             E 
    
    Similar to Traingle, every from face of the diamond's faces are:
        A -> B -> D
        B -> C -> D
        A -> C -> B
        E -> B -> A
        E -> C -> B
        C -> E -> A
    """
    def __init__(self, v_list):
        self.v_list = v_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self):
        l = []
        t_idx = [
                [0,1,3],
                [1,2,3],
                [0,3,2],
                 [4,1,0],
                 [4,2,1],
                 [2,4,0]]
        for i in t_idx:
            l.append(Triangle(self.v_list[i[0]],
                     self.v_list[i[1]], self.v_list[i[2]]))
        return l

    def apply_materials_to_triangles(self):
        for t in self.triangle_list:
            t.set_material(self.ambient, self.diffuse,
                           self.specular, self.shininess, self.reflection)

    def intersect(self, ray: Ray):
        closest_intersection = None
        for triangle in self.triangle_list:
            result = triangle.intersect(ray)
            if result is not None:
                if closest_intersection is None or result[0] < closest_intersection[0]:
                    closest_intersection = result
                    self.last_intersected_triangle = triangle
        return closest_intersection
    
    def compute_normal(self, _):
        return self.last_intersected_triangle.compute_normal(_)

class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        b = self.center - ray.origin
        proj_len = np.dot(b, ray.direction)
        if proj_len < 0:
            return None
        proj = ray.direction * proj_len
        distance = np.linalg.norm(proj - b)
        if distance > self.radius:
            # no intersection
            return None
        elif distance == self.radius:
            # one intersection point
            return proj_len, self
        else:
            # two intersection points
            t = proj_len - np.sqrt(self.radius ** 2 - distance ** 2)
            return t, self
        
    def compute_normal(self, point):
        return normalize(point - self.center)
