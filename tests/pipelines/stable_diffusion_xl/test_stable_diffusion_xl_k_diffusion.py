# coding=utf-8
# Copyright 2023 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import unittest

import numpy as np
import torch

from diffusers import StableDiffusionXLKDiffusionPipeline
from diffusers.utils.testing_utils import enable_full_determinism, nightly, require_torch_gpu, torch_device


enable_full_determinism()


@nightly
@require_torch_gpu
class StableDiffusionPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_stable_diffusion_1(self):
        sdxl_pipe = StableDiffusionXLKDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        sdxl_pipe = sdxl_pipe.to(torch_device)
        sdxl_pipe.set_progress_bar_config(disable=None)

        sdxl_pipe.set_scheduler("sample_euler")

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.manual_seed(0)
        output = sdxl_pipe([prompt], generator=generator, guidance_scale=4.0, num_inference_steps=20, output_type="np")

        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 1024, 1024, 3)
        expected_slice = np.array([0.38376063, 0.3533112, 0.3500121, 0.36708432, 0.36731872, 0.36287862, 0.36724827, 0.35665298, 0.35765022])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_karras_sigmas(self):
        sdxl_pipe = StableDiffusionXLKDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        sdxl_pipe = sdxl_pipe.to(torch_device)
        sdxl_pipe.set_progress_bar_config(disable=None)

        sdxl_pipe.set_scheduler("sample_dpmpp_2m")

        prompt = "A painting of a squirrel eating a burger"
        generator = torch.manual_seed(0)
        output = sdxl_pipe(
            [prompt],
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=15,
            output_type="np",
            use_karras_sigmas=True,
        )

        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 1024, 1024, 3)

        expected_slice = np.array(
            [0.33685943, 0.30347663, 0.23236144, 0.32509226, 0.2707511, 0.23482424, 0.29983392, 0.28047627, 0.24694943]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_noise_sampler_seed(self):
        sdxl_pipe = StableDiffusionXLKDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        sdxl_pipe = sdxl_pipe.to(torch_device)
        sdxl_pipe.set_progress_bar_config(disable=None)

        sdxl_pipe.set_scheduler("sample_dpmpp_sde")

        prompt = "A painting of a squirrel eating a burger"
        seed = 0
        images1 = sdxl_pipe(
            [prompt],
            generator=torch.manual_seed(seed),
            noise_sampler_seed=seed,
            guidance_scale=9.0,
            num_inference_steps=20,
            output_type="np",
        ).images
        images2 = sdxl_pipe(
            [prompt],
            generator=torch.manual_seed(seed),
            noise_sampler_seed=seed,
            guidance_scale=9.0,
            num_inference_steps=20,
            output_type="np",
        ).images

        assert images1.shape == (1, 1024, 1024, 3)
        assert images2.shape == (1, 1024, 1024, 3)
        assert np.abs(images1.flatten() - images2.flatten()).max() < 1e-2
