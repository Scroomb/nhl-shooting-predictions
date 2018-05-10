import numpy as np

def within(p,q,r):
    return p <= q <= r or r <= q <= p

def point_between(a,b,c):
    a = a.reshape(2,)
    b = b.reshape(2,)
    c = c.reshape(2,)
    return ((np.cross((b-a),(c-a)) == 0) and
        (within(a[0], c[0], b[0]) if a[0] != b[0] else within(a[1], c[1], b[1])))

def in_on_triangle(v,v_0,v_1,v_2):
    return in_triangle_bary(v,v_0,v_1,v_2) or point_between(v_0,v_1,v) or point_between(v_0,v_2,v) or point_between(v_1,v_2,v)

def in_triangle_bary(v,v_0,v_1,v_2):
    v0 = v_2 - v_0
    v1 = v_1 - v_0
    v2 = v - v_0
    dot00 = np.dot(v0.T,v0)
    dot01 = np.dot(v0.T,v1)
    dot02 = np.dot(v0.T,v2)
    dot11 = np.dot(v1.T,v1)
    dot12 = np.dot(v1.T,v2)
    invDenom = 1/ (dot00*dot11 - dot01*dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom
    return (u >= 0) & (v >= 0) & (u + v <1 )

def in_triangle_coords(point):
    v_1 = np.array([[89],[39]])
    v_2 = np.array([[89],[45]])
    v_0 = point
    coords = []
    if np.array_equal(v_1,v_0) or np.array_equal(v_2, v_0):
        return None
    for x in range(100):
        for y in range(85):
            v = np.array([[x],[y]])
            if in_triangle_bary(v,v_0,v_1,v_2):
                coords.append((x,y))
    return coords

if __name__ == '__main__':
    v_0 = np.array([[85],[38]])


    v = np.array([[89],[42]])

    # print(in_triangle(v,v_0,v_1,v_2) or point_between(v_0,v_1,v) or point_between(v_0,v_2,v) or point_between(v_1,v_2,v))
    # print(in_on_triangle(v,v_0,v_1,v_2))
    coords = in_triangle_coords(v_0)
