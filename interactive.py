import os
import sys
import numpy as np
from typing import Dict, Tuple
from loguru import logger
from argparse import ArgumentParser

import torch
import torchvision.transforms.functional as TFT

import dearpygui.dearpygui as dpg

from model import Generator


class StyleGAN2Adaptor:
    IMAGE_SIZE = 256
    STYLE_VEC_DIMS = 512

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None
    ) -> None:
        logger.info(f"Building StyleGAN2 ...")
        self._g = Generator(
            self.IMAGE_SIZE,
            self.STYLE_VEC_DIMS,
            8,
            channel_multiplier=2
        )

        logger.info(f"Loading checkpoint from '{checkpoint_path}'")
        self._g.load_state_dict(
            torch.load(checkpoint_path)["g_ema"],
            strict=False
        )

        self._g.eval()

        self._device = device
        if self._device:
            logger.info(f"Moving StyleGAN2 to {self._device}")
            self._g = self._g.to(self._device)

        logger.info(f"Successfully loaded StyleGAN2.")
        pass

    def generate_new_image(self) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            sample_z = torch.randn(1, self.STYLE_VEC_DIMS, device=self._device)
            sample, style_vector = self._g(
                [sample_z],
                truncation=1.0,
                truncation_latent=None,
                return_latents=True,
                randomize_noise=False
            )

        sample = sample.squeeze().cpu().numpy()
        style_vector = style_vector.squeeze().cpu()

        return sample, style_vector

    def regenerate_image(self, style: torch.Tensor):
        with torch.no_grad():
            style = style.to(self._device).unsqueeze(0)
            sample, _ = self._g(
                [style],
                truncation=1.0,
                truncation_latent=None,
                return_latents=True,
                randomize_noise=False,
                input_is_latent=True
            )

        sample = sample.squeeze().cpu().numpy()

        return sample


class StyleGAN2Renderer:
    SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]

    STYLE_VEC_RANGE = (-10, 10)

    def __init__(
        self,
        checkpoint_path: str
    ) -> None:
        self._device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Found device: {self._device}")

        self._style_gan_2 = StyleGAN2Adaptor(
            checkpoint_path,
            self._device
        )

        self._image_buffer = np.ones(
            shape=(
                StyleGAN2Adaptor.IMAGE_SIZE,
                StyleGAN2Adaptor.IMAGE_SIZE,
                4
            ),
            dtype=np.float32
        )

        self._style_vec: torch.Tensor = torch.zeros(
            size=(StyleGAN2Adaptor.STYLE_VEC_DIMS,),
            dtype=torch.float32
        )

        self._default_style_vec: torch.Tensor = torch.zeros(
            size=(StyleGAN2Adaptor.STYLE_VEC_DIMS,),
            dtype=torch.float32
        )
        pass

    def get_image_buffer(self):
        return self._image_buffer

    def get_style_vec_value(self, index: int):
        return self._style_vec[index].item()

    def _update_image_buffer(self):
        dpg.set_value(MainWindow.PYGUI_ID_IMAGE, self._image_buffer)

    def _update_style_vec_values(self):
        for i in range(StyleGAN2Adaptor.STYLE_VEC_DIMS):
            dpg.set_value(
                MainWindow.PYGUI_ID_STYLE_VEC_PRE + str(i),
                self._style_vec[i].item()
            )

    def callback_gen_new_image(self):
        logger.info(f"Generate new image ...")

        image, style = self._style_gan_2.generate_new_image()

        image = (image + 1.0) / 2.0
        self._image_buffer[:, :, :3] = image.transpose(1, 2, 0)

        self._style_vec: torch.Tensor = style[0]
        self._default_style_vec = self._style_vec.clone().detach()

        self._update_image_buffer()
        self._update_style_vec_values()

        logger.info("Done.")
        pass

    def callback_save_image(
        self,
        sender: str,
        app_data: Dict[str, str]
    ):
        logger.info("Save image ...")
        file_path = None

        key_file_path = "file_path_name"
        if key_file_path in app_data.keys():
            file_path = app_data[key_file_path]

        is_OK = file_path is not None

        if is_OK:
            file_path = os.path.abspath(file_path)
            extension = os.path.splitext(file_path)[-1]
            logger.debug(f"extension is '{extension}'")
            is_OK &= (extension in self.SUPPORTED_EXTENSIONS)

            if not is_OK:
                logger.error(
                    f"The extension is not supported. We support {self.SUPPORTED_EXTENSIONS}."
                )

        if is_OK:
            logger.info(f"Saving image to {file_path}")
            image_array = torch.from_numpy(
                self._image_buffer
            ).permute(2, 0, 1).clip(0.0, 1.0)
            image = TFT.to_pil_image(
                image_array
            )
            image.save(file_path)
            logger.info("Done.")
        else:
            logger.error(
                f"Aborted saving the image."
            )
        pass

    def callback_style_vec_value_changed(self, sender: str):
        vector_index = int(sender.split("_")[-1])
        value = dpg.get_value(sender)
        logger.info(f"{vector_index}: value={value}")

        self._style_vec[vector_index] = value

        image = self._style_gan_2.regenerate_image(self._style_vec)

        image = (image + 1.0) / 2.0
        self._image_buffer[:, :, :3] = image.transpose(1, 2, 0)

        self._update_image_buffer()
        pass

    def callback_reset_style_vec(self):
        logger.info(f"Reset style vector values ...")

        self._style_vec = self._default_style_vec.clone().detach()

        image = self._style_gan_2.regenerate_image(self._style_vec)

        image = (image + 1.0) / 2.0
        self._image_buffer[:, :, :3] = image.transpose(1, 2, 0)

        self._update_image_buffer()
        self._update_style_vec_values()

        logger.info("Done.")
        pass


class MainWindow():
    MAIN_WINDOW_TITLE = "StyleGAN2"
    MAIN_WINDOW_HEIGHT = 1000
    MAIN_WINDOW_WIDTH = 1800

    IMAGE_SYN_WINDOW_TITLE = "Image Synthesis"
    IMAGE_SYN_WINDOW_H_PROP = 0.1
    IMAGE_SYN_WINDOW_W_PROP = 0.25

    STYLE_VEC_WINDOW_TITLE = "Style Vectors"

    IMAGE_WINDOW_TITLE = "Generated Image"

    PYGUI_ID_SAVE_IMAGE_FILE_SELECTOR = "save_image_file_dialog_id"
    PYGUI_ID_IMAGE = "image_id"
    PYGUI_ID_IMAGE_WINDOW = "image_window_id"
    PYGUI_ID_STYLE_VEC_PRE = "style_vec_"

    def __init__(self, checkpoint_path: str) -> None:
        self._stylegan2_renderer = StyleGAN2Renderer(
            checkpoint_path
        )

        self._build_ui()

    def _build_ui(self):
        dpg.create_context()
        dpg.create_viewport(
            title=self.MAIN_WINDOW_TITLE,
            width=self.MAIN_WINDOW_WIDTH,
            height=self.MAIN_WINDOW_HEIGHT
        )
        dpg.setup_dearpygui()

        img_syn_window_height = int(
            self.MAIN_WINDOW_HEIGHT * self.IMAGE_SYN_WINDOW_H_PROP
        )
        img_syn_window_width = int(
            self.MAIN_WINDOW_WIDTH * self.IMAGE_SYN_WINDOW_W_PROP
        )
        style_vec_window_height = int(
            self.MAIN_WINDOW_HEIGHT - img_syn_window_height - 50
        )
        style_vec_window_width = img_syn_window_width

        def cancel_callback(sender, app_data):
            logger.info('Canceled.')

        # with dpg.file_dialog(
        #     directory_selector=False,
        #     show=False,
        #     callback=self._stylegan2_renderer.callback_save_image,
        #     cancel_callback=cancel_callback,
        #     id=self.PYGUI_ID_SAVE_IMAGE_FILE_SELECTOR,
        #     width=700,
        #     height=400
        # ):
        #     for ext in StyleGAN2Renderer.SUPPORTED_EXTENSIONS:
        #         dpg.add_file_extension(ext)

        with dpg.window(
            label=self.IMAGE_SYN_WINDOW_TITLE,
            min_size=(
                img_syn_window_width,
                img_syn_window_height
            ),
            max_size=(
                img_syn_window_width,
                img_syn_window_height
            ),
            no_resize=True,
            no_move=True,
            no_close=True,
            no_collapse=True
        ) as img_syn_window:
            dpg.add_text("Hello world")
            dpg.add_button(
                label="Generate new image",
                callback=self._stylegan2_renderer.callback_gen_new_image
            )
            dpg.add_button(
                label="Save image",
                callback=lambda: dpg.show_item(
                    self.PYGUI_ID_SAVE_IMAGE_FILE_SELECTOR
                )
            )
            pass

        with dpg.window(
            label=self.STYLE_VEC_WINDOW_TITLE,
            min_size=(
                style_vec_window_width,
                style_vec_window_height
            ),
            max_size=(
                style_vec_window_width,
                style_vec_window_height
            ),
            # autosize=True,
            no_resize=True,
            no_move=True,
            no_close=True,
            no_collapse=True,
            pos=(0, img_syn_window_height)
        ) as style_vectors_window:
            dpg.add_button(
                label="Reset values",
                callback=self._stylegan2_renderer.callback_reset_style_vec
            )
            for iVec in range(StyleGAN2Adaptor.STYLE_VEC_DIMS):
                dpg.add_slider_float(
                    label=f"Style Vector {iVec}",
                    default_value=self._stylegan2_renderer.get_style_vec_value(
                        iVec
                    ),
                    tag=self.PYGUI_ID_STYLE_VEC_PRE + f"{iVec}",
                    min_value=StyleGAN2Renderer.STYLE_VEC_RANGE[0],
                    max_value=StyleGAN2Renderer.STYLE_VEC_RANGE[1],
                    callback=self._stylegan2_renderer.callback_style_vec_value_changed,
                    drop_callback=self._stylegan2_renderer.callback_style_vec_value_changed
                )

        with dpg.texture_registry():
            image_buffer = self._stylegan2_renderer.get_image_buffer()
            dpg.add_dynamic_texture(
                width=image_buffer.shape[0],
                height=image_buffer.shape[1],
                default_value=image_buffer,
                tag=self.PYGUI_ID_IMAGE
            )

        with dpg.window(
            label=self.IMAGE_WINDOW_TITLE,
            min_size=(
                int(StyleGAN2Adaptor.IMAGE_SIZE * 1.2),
                int(StyleGAN2Adaptor.IMAGE_SIZE * 1.2)
            ),
            max_size=(
                int(StyleGAN2Adaptor.IMAGE_SIZE * 1.2),
                int(StyleGAN2Adaptor.IMAGE_SIZE * 1.2)
            ),
            # autosize=True,
            no_resize=True,
            no_move=True,
            no_close=True,
            no_collapse=True,
            pos=(img_syn_window_width, 0)
        ) as parent_window02:
            dpg.add_image(self.PYGUI_ID_IMAGE, tag=self.PYGUI_ID_IMAGE_WINDOW)

        logger.info("Launching window!")
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()


def parse_args():
    parser = ArgumentParser(
        prog="An interactive visualizer to dive into the style space of StyleGAN2."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint/550000.pt"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=[
            "TRACE",
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL"
        ]
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    logger.remove()
    logger.add(
        sys.stdout,
        level=args.log_level
    )

    window = MainWindow(
        checkpoint_path=args.checkpoint
    )

    pass


if __name__ == "__main__":
    main()
