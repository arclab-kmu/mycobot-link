"""myCobot 280 FK / numeric IK (standard DH, mm, deg). Validated against firmware get_coords."""
import numpy as np
LO = np.array([-164.0, -136.0, -146.0, -146.0, -151.0, -176.0]); HI = np.array([164.0, 136.0, 146.0, 146.0, 156.0, 176.0])
DH = [  # theta_offset, d, a, alpha
    (0.0,        139.42,   0.0,   np.pi/2),
    (-np.pi/2,   0.0,   -110.4,   0.0),
    (0.0,        0.0,    -96.0,   0.0),
    (-np.pi/2,   63.4,     0.0,   np.pi/2),
    (np.pi/2,    75.05,    0.0,  -np.pi/2),
    (0.0,        45.6,     0.0,   0.0),
]
def _A(th, d, a, al):
    ct, st, ca, sa = np.cos(th), np.sin(th), np.cos(al), np.sin(al)
    return np.array([[ct, -st*ca, st*sa, a*ct], [st, ct*ca, -ct*sa, a*st], [0, sa, ca, d], [0, 0, 0, 1]])
def fk(q_deg, tool=np.eye(4)):
    T = np.eye(4)
    for qi, (off, d, a, al) in zip(np.radians(q_deg), DH):
        T = T @ _A(qi + off, d, a, al)
    return T @ tool
def rpy_xyz(R):  # firmware rx, ry, rz (deg), R = Rz(rz) Ry(ry) Rx(rx)
    ry = np.arcsin(-R[2, 0]); rx = np.arctan2(R[2, 1], R[2, 2]); rz = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees([rx, ry, rz])
def ik(target_T, q0, w_rot=60.0, iters=2000, tool=np.eye(4), lock=()):
    """Damped least squares on [pos mm, rot*w_rot]. Returns q (deg), pos err mm, rot err deg."""
    q = np.radians(np.array(q0, float))
    for _ in range(iters):
        T = fk(np.degrees(q), tool)
        dp = target_T[:3, 3] - T[:3, 3]
        Re = target_T[:3, :3] @ T[:3, :3].T
        ang = np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
        dr = np.zeros(3) if ang < 1e-9 else ang / (2*np.sin(ang)) * np.array([Re[2,1]-Re[1,2], Re[0,2]-Re[2,0], Re[1,0]-Re[0,1]])
        e = np.concatenate([dp, w_rot*dr])
        if np.linalg.norm(dp) < 0.2 and np.degrees(ang) < 0.2: break
        J = np.zeros((6, 6)); h = 1e-4
        for j in range(6):
            if j in lock: continue
            qq = q.copy(); qq[j] += h; T2 = fk(np.degrees(qq), tool)
            dp2 = T2[:3, 3] - T[:3, 3]
            Rd = T2[:3, :3] @ T[:3, :3].T
            dr2 = np.array([Rd[2,1]-Rd[1,2], Rd[0,2]-Rd[2,0], Rd[1,0]-Rd[0,1]]) / 2
            J[:, j] = np.concatenate([dp2, w_rot*dr2]) / h
        lam = 0.5 + 0.05*np.linalg.norm(e)
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2*np.eye(6), e)
        q += np.clip(dq, -0.03, 0.03)
        q = np.clip(q, np.radians(LO), np.radians(HI))
    T = fk(np.degrees(q), tool)
    Re = target_T[:3, :3] @ T[:3, :3].T
    return np.degrees(q), float(np.linalg.norm(target_T[:3, 3]-T[:3, 3])), float(np.degrees(np.arccos(np.clip((np.trace(Re)-1)/2, -1, 1))))
