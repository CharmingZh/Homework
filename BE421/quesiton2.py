import cv2
import numpy as np
import matplotlib.pyplot as plt

# load original img
image = cv2.imread('jig.jpeg')
# OpenCV uses BGR in default, chenge it into RGB for plt
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 3D coord, some points are counted by the relationship between other points
world_coordinates = np.array([
    [0, 0, 0],  # A
    [0, 6, 0],  # B
    [11, 6, 0],  # C
    [11, 0, 0],  # D
    [8.25, 0, -4.5],  # E
    [2.75, 0, -4.5],  # F
    [5.5, 0, -3.5],  # G
    [5.5, 6, -3.5],  # H
    [2, 0, 0],  # K
    [2, 6, 0],  # L
    [9, 6, 0],  # M
    [9, 0, 0],  # N
    [8.25, 0, -1.8125],  # O
    [2.75, 0, -1.8125]  # P
])

# 2D coord, were extracted using a custom tool I developed, which allows users to click on points and outputs their coordinates.
image_coordinates = np.array([
    [63, 166],  # a
    [126, 95],  # b
    [543, 97],  # c
    [516, 172],  # d
    [406, 354],  # e
    [184, 346],  # f
    [291, 311],  # g
    [336, 226],  # h
    [141, 169],  # k
    [199, 96],  # l
    [463, 96],  # m
    [430, 174],  # n
    [402, 247],  # o
    [179, 242]  # p
])

def build_A_matrix(world_points, image_points):
    A = []
    for i in range(len(world_points)):
        X, Y, Z = world_points[i]
        x, y = image_points[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y])
    return np.array(A)


def estimate_camera_matrix(world_points, image_points):
    A = build_A_matrix(world_points, image_points)
    # SVD
    m, n = A.shape
    U = np.zeros((m, m))
    S = np.zeros(min(m, n))
    Vt = np.zeros((n, n))

    # eigen value and vectors of A^T * A
    AtA = A.T @ A
    eigenvalues, eigenvectors = np.linalg.eigh(AtA)

    # sort the eigen value and vector
    sorted_indices = np.argsort(eigenvalues)[::-1]
    S = np.sqrt(np.maximum(eigenvalues[sorted_indices], 0))
    Vt = eigenvectors[:, sorted_indices].T

    # matrix U
    inv_S = np.diag(1 / S[S > 1e-10])
    U = A @ Vt.T @ inv_S.T

    # get matrix P
    P = Vt[-1].reshape(3, 4)
    return P


def project_points(world_points, P):
    ones = np.ones((world_points.shape[0], 1))
    world_points_homogeneous = np.hstack((world_points, ones))
    image_points_projected = P @ world_points_homogeneous.T
    image_points_projected /= image_points_projected[2, :]
    return image_points_projected[:2, :].T


def calculate_reprojection_error(observed_points, projected_points):
    errors = np.linalg.norm(observed_points - projected_points, axis=1)
    mean_error = np.mean(errors)
    return mean_error, errors


def visualize_on_image_with_labels(image, observed_points, projected_points, errors, point_labels):
    plt.figure(figsize=(10, 8), dpi=1000)
    plt.imshow(image)

    plt.scatter(observed_points[:, 0], observed_points[:, 1], color='green', s=3, label='Observed Points', zorder=2)

    plt.scatter(projected_points[:, 0], projected_points[:, 1], color='red', s=3, label='Projected Points', zorder=2)

    for i in range(len(observed_points)):
        plt.plot([observed_points[i, 0], projected_points[i, 0]],
                 [observed_points[i, 1], projected_points[i, 1]],
                 'k--', linewidth=1, alpha=0.5, zorder=1)

    # 显示误差的大小和点的名称
    for i, (error, label) in enumerate(zip(errors, point_labels)):
        plt.text(observed_points[i, 0], observed_points[i, 1], f'{label} ({error:.2f})', fontsize=14, color='dodgerblue', zorder=3)

    plt.title('Re-projected Error Visualization with Point Labels')
    plt.xlabel('col')
    plt.ylabel('row')
    plt.legend()
    plt.tight_layout()
    plt.savefig('reprojected_error_visualization.png', bbox_inches='tight')
    plt.show()


def visualize_error_bars(errors, point_labels):
    plt.figure(figsize=(10, 6), dpi=1000)

    plt.bar(point_labels, errors, color='orange')
    plt.xlabel('Points')
    plt.ylabel('Re-projected Error (pixels)')
    plt.title('Re-projected Error for Each Point')

    for i, error in enumerate(errors):
        plt.text(i, error + 0.1, f'{error:.2f}', ha='center', va='bottom')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reprojected_error_bars.png', bbox_inches='tight')
    plt.show()


P = estimate_camera_matrix(world_coordinates, image_coordinates)

projected_points = project_points(world_coordinates, P)

mean_error, errors = calculate_reprojection_error(image_coordinates, projected_points)

print("Estimated Camera Matrix (3x4):\n", P)
print(f"\nMean reprojection error: {mean_error:.4f}")

point_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'h', 'K', 'L', 'M', 'N', 'O', 'P']
visualize_on_image_with_labels(image_rgb, image_coordinates, projected_points, errors, point_labels)

mean_error, errors = calculate_reprojection_error(image_coordinates, projected_points)

visualize_error_bars(errors, point_labels)
