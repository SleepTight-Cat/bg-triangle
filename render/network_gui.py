import time

import viser
import viser.transforms as tf

import numpy as np
host = "0.0.0.0"
port = 8080

server = {
    "server": None,
    "render_type": None,
    "slider_boundary_scale": None,
    "button_optimize": None,
    "button_optimize_state": None,
}


class dummygui:
    def __init__(self, value):
        self.value = value


def init(wish_host, wish_port, pipeline_args):
    global host, port, server
    host = wish_host
    port = wish_port

    server["server"] = viser.ViserServer(host=host, port=port)
    server["server"].scene.world_axes.visible = True
    server["server"].scene.set_up_direction(direction = '+z')




    server["render_type"] = dummygui("debug")
    server["num_segments_per_bprimitive_edge"] = dummygui(pipeline_args.num_segments_per_bprimitive_edge)

    server["slider_boundary_scale"]= dummygui(pipeline_args.log_blur_radius)
    server["gaussian_scale"]= dummygui(pipeline_args.scale_gaussian)
    server["learning_rate"]= dummygui(1.0)
    server["mode"]= dummygui(1)
    server["button_train"]= dummygui(True)
    server["button_train_state"]= True


    server["button_optimize"] = dummygui(False)
    server["button_optimize_state"] = True



    server["checkbox_split"] = dummygui(True)

    server["checkbox_opt_position"] = dummygui(False)

    server["checkbox_pts_mode"] = dummygui(False)


    @server["server"].on_client_connect
    def _(client: viser.ClientHandle) -> None:

        # This will run whenever we get a new camera!
        #@client.camera.on_update
        #def _(_: viser.CameraHandle) -> None:
        #    print(f"New camera on client {client.client_id}!")




        # Show the client ID in the GUI.
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True

        global server
        server["render_type"] = client.gui.add_dropdown(
            "Render Type",
            options=[
                "Depth Map",
                "Segmentation",
                "Colored UVW",
                "Colored Boundary Points",
                "debug",
                "GT Image",
                "Surface Normal",
                "Depth Normal",
                "Accumulated Gradient Image",
                "Accum_Edge",
                "Accum_Vis",
                "edges",
            ],
            initial_value="debug"
        )

        server["num_segments_per_bprimitive_edge"] = client.gui.add_slider(
            "# Segments/BP",
            min=1,
            max=10,
            step=1,
            initial_value=pipeline_args.num_segments_per_bprimitive_edge
        )

        server["slider_boundary_scale"] = client.gui.add_slider(
            "Scale Boundary",
            min=-9.0,
            max=-2.0,
            step=0.1,
            initial_value=pipeline_args.log_blur_radius
        )

        server["gaussian_scale"] = client.gui.add_slider(
            "Scale Gaussian",
            min=0.05,
            max=1.3,
            step=0.05,
            initial_value=pipeline_args.scale_gaussian
        )

        server["learning_rate"] = client.gui.add_slider(
            "Learning Rate",
            min=0.1,
            max=1.2,
            step=0.05,
            initial_value=1.0
        )


        server["mode"] = client.gui.add_slider(
            "Training Mode",
            min=1, # Having point cloud initialization, we don't need mode 0 any more.......
            max=2,
            step=1,
            initial_value=1
        )

        server["button_train"] = client.gui.add_button(
            "Stop Train",
        )
        server["button_train_state"] = True

        server["button_optimize"] = client.gui.add_button(
            "Stop Backward",
        )
        server["button_optimize_state"] = True


        gui_reset_up = client.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
        )

        @gui_reset_up.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None
            client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                [0.0, -1.0, 0.0]
            )

        server["checkbox_split"] = client.gui.add_checkbox(
            "split", True
        )
        server["checkbox_opt_position"] = client.gui.add_checkbox(
            "Opt Position", False
        )
        server["checkbox_pts_mode"] = client.gui.add_checkbox(
            "Point Mode", False
        )



    return server

def try_connect():
    if server["server"] is None:
        return

    while True:
        # Get all currently connected clients.
        clients = server["server"].get_clients()
        print("Connected client IDs", clients.keys())

        if len(clients) == 0:
            time.sleep(0.5)
        else:
            server["client"] = clients[0]
            break

def on_gui_change():
    if server["server"] is None:
        return ""

    if server["button_train"].value:
        if server["button_train_state"]:
            server["button_train"].label = "Start Train"
        else:
            server["button_train"].label = "Stop Train"
        server["button_train_state"] = not server["button_train_state"]
        server["button_train"].value = False

    if server["button_optimize"].value:
        if server["button_optimize_state"]:
            server["button_optimize"].label = "Do Optimization"
        else:
            server["button_optimize"].label = "Stop Optimization"
        server["button_optimize_state"] = not server["button_optimize_state"]
        server["button_optimize"].value = False

    return server["render_type"].value
