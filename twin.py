import numpy as np
import networkx as nx
from pathlib import Path

try:
    import pyvista as pv
    pv.OFF_SCREEN = True
except ImportError:
    pv = None

class DT04_3DVolume:
    @staticmethod
    def generate_volume(output_path: Path) -> bool:
        if pv is None: return False
        
        grid = pv.ImageData()
        grid.dimensions = np.array([31, 19, 8])
        grid.spacing = (1.0, 1.0, 1.0)
        grid.origin = (0.0, 0.0, 0.0)
        
        points = grid.points
        z_norm = points[:, 2] / 7.0
        
        f_values = 0.9 - (z_norm * 0.4) 
        dist_to_auditorio = np.linalg.norm(points[:, :2] - np.array([15.0, 10.0]), axis=1)
        f_values -= np.where(dist_to_auditorio < 5.0, 0.3, 0.0)
        grid.point_data["F_global"] = np.clip(f_values, 0.01, 1.0)
        
        isosurfaces = grid.contour([0.5, 0.7, 0.9], scalars="F_global")
        grid.save(output_path / "horse_cft_volume.vtk")
        
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background("#1a1a1a")
        plotter.add_mesh(grid.outline(), color="#3a3a3c", line_width=2)
        plotter.add_mesh(isosurfaces, scalars="F_global", cmap=["#ff453a", "#ff9f0a", "#30d158"], opacity=0.6)
        plotter.camera_position = 'iso'
        plotter.screenshot(output_path / "3d_building.png")
        plotter.close()
        return True

class DT09_BuildingGraph:
    @staticmethod
    def create_horse_cft_graph() -> nx.Graph:
        G = nx.Graph()
        edges = [
            ("Auditorio", "Circ_GF", 3.0), 
            ("Circ_GF", "Hall", 2.0), 
            ("Hall", "EXIT_GF_Primary", 1.0), 
            ("Sala_A", "Circ_GF", 2.0),
            ("Dojo", "Circ_FF", 2.0),
            ("Circ_FF", "Circ_GF", 4.0)
        ]
        G.add_weighted_edges_from(edges)
        return G
