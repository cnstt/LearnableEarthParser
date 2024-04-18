import os
import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from imageio import get_writer
from tqdm import tqdm
import argparse


class Renderer:
    """ Render video of the training process, by using the data logged at each step in the tensorboard log files.
    This enables the user to visualize the convergence of the training, for the scene and the prototypes.
    """
    def __init__(self, args):
        self.log_dir = args.log_dir
        self.temp_dir = args.temp_dir
        self.keep_first_gt = args.keep_first_gt
        self.fps = args.fps
        self.max_step = args.max_step
        self.origin_lidar = args.origin_lidar
        self.report = args.report
        self.protodisplay = args.protodisplay

        self.mesh_events = self.get_events()
        self.fig, self.ax, self.fig_protos, self.ax_protos, self.gt_pos, self.gt_color = self.init_render()
    
    def get_events(self):  
        # Find all event files in the directory
        event_files = tf.io.gfile.glob(self.log_dir + '/events*')

        if not event_files:
            raise Exception("No event files found in the specified directory.")
        else:
            # Retrieve all the events from the event file
            mesh_events = []
            for event in tf.compat.v1.train.summary_iterator(event_files[0]):
                step = event.step if hasattr(event, 'step') else None
                
                vertex_array = None
                color_array = None
                for value in event.summary.value:
                    if value.tag.startswith("pred_train_VERTEX"):
                        vertex_array = tf.make_ndarray(value.tensor)
                    elif value.tag.startswith("pred_train_COLOR"):
                        color_array = tf.make_ndarray(value.tensor)
                    elif value.tag.startswith("protos_pointcloud_VERTEX"):
                        protos_array = tf.make_ndarray(value.tensor)
                    elif value.tag.startswith("protos_pointcloud_COLOR"):
                        protosC_array = tf.make_ndarray(value.tensor)     
                if vertex_array is not None and color_array is not None:
                    mesh_events.append((step, vertex_array, color_array, protos_array, protosC_array))
            return mesh_events
    
    # Update the plot for each step
    def update_plot(self, step, scene_vertices, scene_colors, protos, protosC):
        self.ax.clear()
        vertices = scene_vertices[0]
        colors = scene_colors[0].astype(np.uint8)
        indices = np.argwhere((colors[:, 1] == 0) & (colors[:, 2] == 0)).flatten()
        min_index = np.min(indices)
        sub_mean = self.origin_lidar - vertices[-1688]
        vertices[:,:] += sub_mean
        if self.keep_first_gt:
            vertices = np.concatenate((self.gt_pos, vertices[min_index:]), axis=0)
            colors = np.concatenate((self.gt_color, colors[min_index:,:3]), axis=0)
        self.ax.scatter(vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                c=colors / 255.0,
                s=2,
                depthshade=False)
        
        self.ax.set_box_aspect(aspect=(0.8125, 1.0, .1875))
        self.ax.set_xlim3d([6, 32])
        self.ax.set_ylim3d([0, 32])
        self.ax.set_zlim3d([0, 6])
        self.ax.set_zticks([0, 3, 6])
        # Adjust the camera settings
        self.ax.view_init(elev=45)  # Set the elevation and azimuth angles
        self.ax.dist = 5  # Set the zoom level (you can adjust this value)

        self.ax.set_title(f"Step: {step}")
        self.ax.set_proj_type('persp')
        
        # Plot protos 1-3 vertically
        for i in range(3):
            protos_vertices = protos[i]
            protos_colors = protosC[i]/255.
            self.ax_protos[i].clear()
            self.ax_protos[i].tick_params(which='major', labelsize=7, pad=-3.5)
            self.ax_protos[i].view_init(elev=45)
            self.ax_protos[i].scatter(protos_vertices[:, 0],
                                protos_vertices[:, 1],
                                protos_vertices[:, 2],
                                c=protos_colors,
                                s=2,
                                depthshade=False)
        if self.protodisplay:
            title0 = self.ax_protos[0].set_title("Cube")
            title0.set_position([-0.3, -1])
            title1 = self.ax_protos[1].set_title("Cylinder")
            title1.set_position([-0.3, -1])
            title2 = self.ax_protos[2].set_title("Sphere")
            title2.set_position([-0.3, -1])
    
    def init_render(self):
        # Create the initial plotax
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.set_proj_type('persp')

        fig_protos, ax_protos = plt.subplots(3, figsize=(4, 6), subplot_kw={'projection': '3d'})
        fig_protos.suptitle('Protos 1-3')

        colors = self.mesh_events[0][2][0]
        pos = self.mesh_events[0][1][0]
        # Renormalise pose
        sub_mean = self.origin_lidar - pos[-1688]
        pos[:,:] += sub_mean
        indices = np.argwhere((colors[:, 1] == 0) & (colors[:, 2] == 0)).flatten()
        init_min_index = np.min(indices)
        # print(init_min_index)
        gt_pos = pos[:init_min_index]
        gt_color = colors[:init_min_index]

        ax.set_box_aspect(aspect=(0.8125, 1, 0.1875))
        ax.set_xlim3d(6, 32)
        ax.set_ylim3d(0, 32)
        ax.set_zlim3d(0, 6)
        ax.set_zticks([0, 3, 6])
        # Adjust the camera settings
        ax.view_init(elev=45)  # Set the elevation and azimuth angles
        ax.dist = 5  # Set the zoom level (you can adjust this value)
        return fig, ax, fig_protos, ax_protos, gt_pos, gt_color
    
    def exec(self):
        # Create the animation frames
        frames = []
        frames_protos = []
        for step, vertex_array, color_array, protos_array, protosC_array in tqdm(self.mesh_events, desc="Creating Frames", unit="step"):
            self.update_plot(step,
                            vertex_array,
                            color_array,
                            protos_array,
                            protosC_array)
            if "scene" in self.report:
                buf = io.BytesIO()
                self.fig.savefig(buf, format='png', dpi=300)
                buf.seek(0)
                frames.append(np.array(plt.imread(buf)))
            if "proto" in self.report:
                self.fig_protos.suptitle(f"Step: {step}")
                buf = io.BytesIO()
                self.fig_protos.savefig(buf, format='png', dpi=300)
                buf.seek(0)
                frames_protos.append(np.array(plt.imread(buf)))
            if step >= self.max_step:
                break

        # Save the frames as a video
        video_path = os.path.expanduser(os.path.join(self.temp_dir, "mesh_evolution_with_color.mp4"))
        # Check if the file already exists
        counter = 1
        while os.path.exists(video_path):
            # Modify the filename by adding a counter
            video_path = os.path.expanduser(os.path.join(self.temp_dir, f"mesh_evolution_with_color_{counter}.mp4"))
            counter += 1

        # Save the frames as a video
        videoProto_path = os.path.expanduser(os.path.join(self.temp_dir, "proto_evolution_with_color.mp4"))
        # Check if the file already exists
        counter = 1
        while os.path.exists(videoProto_path):
            # Modify the filename by adding a counter
            videoProto_path = os.path.expanduser(os.path.join(self.temp_dir, f"proto_evolution_with_color_{counter}.mp4"))
            counter += 1

        if "scene" in self.report:
            with get_writer(video_path, fps=self.fps) as writer:
                for frame in frames:
                    # Ensure frame pixel values are in uint8 range
                    frame_uint8 = (frame * 255).astype(np.uint8)
                    writer.append_data(frame_uint8)
            print(f"Video saved to: {video_path}")
        if "proto" in self.report:
            with get_writer(videoProto_path, fps=self.fps) as writer:
                for frame in frames_protos:
                    frame_uint8 = (frame * 255).astype(np.uint8)
                    writer.append_data(frame_uint8)
            print(f"Video (protos) saved to: {videoProto_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Render video from TensorBoard event files.")
    parser.add_argument("log_dir", type=str, help="Path to the directory containing the TensorBoard event file")
    parser.add_argument("--temp_dir", type=str, default="~/outputs/videos", help="Directory to store temporary video files")
    parser.add_argument("--keep_first_gt", action="store_true", help="Keep the first ground truth")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the output video")
    parser.add_argument("--max_step", type=int, default=10000, help="Maximum step for rendering")
    parser.add_argument("--origin_lidar", nargs="+", type=float, default=[16, 16, 6], help="Origin lidar coordinates")
    parser.add_argument("--report", type=str, default="scene+proto", help="Report type ('scene', 'proto', or 'scene+proto')")
    parser.add_argument("--protodisplay", action="store_true", help="Display protos")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    renderer = Renderer(args)
    renderer.exec()
