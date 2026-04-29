from collections import namedtuple
from copy import copy
from itertools import permutations, chain
import random
import csv
import os.path
from io import StringIO
from PIL import Image
import numpy as np

import modules.scripts as scripts
import gradio as gr
from modules import script_callbacks, images, sd_samplers, processing, sd_models, sd_vae, sd_schedulers, errors
from modules.processing import process_images, Processed, StableDiffusionProcessingTxt2Img
from modules.shared import opts, state
import modules.shared as shared
import re
from modules.ui_components import ToolButton, InputAccordion
import importlib
import traceback

# UI用アイコン
fill_values_symbol = "\U0001f4d2"  # 📒

# --- ▼▼▼ 改造版の関数定義 ▼▼▼ ---

def draw_xyz_grid_unified(p, xs, ys, zs, x_labels, y_labels, z_labels, cell, margin_size, draw_legend_x, draw_legend_y, draw_legend_z, draw_legend_batch, include_sub_grids, split_grids_by_batch):
    batch_size = p.batch_size
    num_cells_total = len(xs) * len(ys) * len(zs)
    state.job_count = num_cells_total * p.n_iter
    
    # 全セルの「純粋な個別画像とメタデータ」を保持するリスト
    all_cell_images = []
    all_cell_prompts = []
    all_cell_seeds = []
    all_cell_subseeds = []
    all_cell_negative_prompts = []
    all_cell_infotexts = []

    processed_template = None

    # --- 1. 生成ループ：全セルの画像をひたすら集める ---
    for iz, z in enumerate(zs):
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                state.job = f"{ix + iy * len(xs) + iz * len(xs) * len(ys) + 1} out of {num_cells_total}"
                processed_cell_result = cell(x, y, z, ix, iy, iz)

                if processed_template is None:
                    if not processed_cell_result.images:
                        raise RuntimeError("The first cell failed to produce any images.")
                    processed_template = copy(processed_cell_result)

                if processed_cell_result.images:
                    # プレビュー画像等の混入を避けるため純粋な生成画像のみを抽出
                    idx_first = getattr(processed_cell_result, 'index_of_first_image', 0)
                    lone_imgs = processed_cell_result.images[idx_first:]
                    n = len(lone_imgs)
                    
                    def get_meta(lst, default_val):
                        if not lst: return [default_val] * n
                        res = lst[idx_first:idx_first+n]
                        if len(res) < n:
                            res.extend([lst[-1] if lst else default_val] * (n - len(res)))
                        return res

                    all_cell_images.extend(lone_imgs)
                    all_cell_infotexts.extend(get_meta(processed_cell_result.infotexts, getattr(processed_cell_result, 'info', "")))
                    all_cell_prompts.extend(get_meta(processed_cell_result.all_prompts, getattr(processed_cell_result, 'prompt', "")))
                    all_cell_seeds.extend(get_meta(processed_cell_result.all_seeds, getattr(processed_cell_result, 'seed', -1)))
                    all_cell_subseeds.extend(get_meta(getattr(processed_cell_result, 'all_subseeds', []), getattr(processed_cell_result, 'subseed', -1)))
                    all_cell_negative_prompts.extend(get_meta(getattr(processed_cell_result, 'all_negative_prompts', []), getattr(processed_cell_result, 'negative_prompt', "")))

    if not all_cell_images:
        return Processed(p, [])

    final_grids = []
    final_grids_infos = []

    if split_grids_by_batch and batch_size > 1:
        # ===== バッチ分割ONの場合 =====
        for batch_idx in range(batch_size):
            batch_images = []
            batch_infos = []
            for cell_idx in range(num_cells_total):
                idx = (cell_idx * batch_size) + batch_idx
                if idx < len(all_cell_images):
                    batch_images.append(all_cell_images[idx])
                    batch_infos.append(all_cell_infotexts[idx])

            if not batch_images:
                continue

            if len(zs) > 1:
                z_slice_grids = []
                for iz in range(len(zs)):
                    start = iz * len(xs) * len(ys)
                    end = start + len(xs) * len(ys)
                    slice_imgs = batch_images[start:end]
                    sub_grid = images.image_grid(slice_imgs, rows=len(ys))
                    if draw_legend_x or draw_legend_y:
                        hor_texts = [[images.GridAnnotation(x)] for x in x_labels] if draw_legend_x else [[images.GridAnnotation()] for _ in x_labels]
                        ver_texts = [[images.GridAnnotation(y)] for y in y_labels] if draw_legend_y else [[images.GridAnnotation()] for _ in y_labels]
                        grid_max_w, grid_max_h = map(max, zip(*(img.size for img in slice_imgs)))
                        sub_grid = images.draw_grid_annotations(sub_grid, grid_max_w, grid_max_h, hor_texts, ver_texts, margin_size)
                    z_slice_grids.append(sub_grid)

                z_grid = images.image_grid(z_slice_grids, rows=1)
                if draw_legend_z or draw_legend_batch:
                    z_texts = [[images.GridAnnotation(z)] for z in z_labels] if draw_legend_z else [[images.GridAnnotation()] for _ in z_labels]
                    batch_texts = [[images.GridAnnotation(f"Batch #{batch_idx+1}")]] if draw_legend_batch else [[images.GridAnnotation()]]
                    grid_max_w, grid_max_h = map(max, zip(*(img.size for img in z_slice_grids)))
                    z_grid = images.draw_grid_annotations(z_grid, grid_max_w, grid_max_h, z_texts, batch_texts)
                
                final_grids.append(z_grid)
                final_grids_infos.append(batch_infos[0] if batch_infos else "")
            else:
                grid = images.image_grid(batch_images, rows=len(ys))
                if draw_legend_x or draw_legend_y:
                    hor_texts = [[images.GridAnnotation(x)] for x in x_labels] if draw_legend_x else [[images.GridAnnotation()] for _ in x_labels]
                    ver_texts = [[images.GridAnnotation(y)] for y in y_labels] if draw_legend_y else [[images.GridAnnotation()] for _ in y_labels]
                    grid_max_w, grid_max_h = map(max, zip(*(img.size for img in batch_images)))
                    grid = images.draw_grid_annotations(grid, grid_max_w, grid_max_h, hor_texts, ver_texts, margin_size)
                
                final_grids.append(grid)
                final_grids_infos.append(batch_infos[0] if batch_infos else "")

        # 最終結果の組み立て（グリッド ＋ 全個別画像）
        final_processed = processed_template
        final_processed.images = final_grids + all_cell_images
        
        num_grids = len(final_grids)
        fallback_info = all_cell_infotexts[0] if all_cell_infotexts else ""
        fallback_prompt = all_cell_prompts[0] if all_cell_prompts else ""
        fallback_seed = all_cell_seeds[0] if all_cell_seeds else -1

        final_processed.infotexts = ([fallback_info] * num_grids) + all_cell_infotexts
        final_processed.all_prompts = ([fallback_prompt] * num_grids) + all_cell_prompts
        final_processed.all_seeds = ([fallback_seed] * num_grids) + all_cell_seeds
        final_processed.all_subseeds = ([-1] * num_grids) + all_cell_subseeds
        final_processed.all_negative_prompts = ([""] * num_grids) + all_cell_negative_prompts
        
        final_processed.index_of_first_image = num_grids # ギャラリーの分割点

        return final_processed

    else:
        # ===== バッチ分割OFFの場合 (標準XYZと同じ挙動) =====
        batch_images = []
        for cell_idx in range(num_cells_total):
            idx = cell_idx * batch_size
            if idx < len(all_cell_images):
                batch_images.append(all_cell_images[idx])
        
        if len(zs) > 1:
            z_slice_grids = []
            for iz in range(len(zs)):
                start = iz * len(xs) * len(ys)
                end = start + len(xs) * len(ys)
                slice_imgs = batch_images[start:end]
                sub_grid = images.image_grid(slice_imgs, rows=len(ys))
                if draw_legend_x or draw_legend_y:
                    hor_texts = [[images.GridAnnotation(x)] for x in x_labels] if draw_legend_x else [[images.GridAnnotation()] for _ in x_labels]
                    ver_texts = [[images.GridAnnotation(y)] for y in y_labels] if draw_legend_y else [[images.GridAnnotation()] for _ in y_labels]
                    grid_max_w, grid_max_h = map(max, zip(*(img.size for img in slice_imgs)))
                    sub_grid = images.draw_grid_annotations(sub_grid, grid_max_w, grid_max_h, hor_texts, ver_texts, margin_size)
                z_slice_grids.append(sub_grid)

            z_grid = images.image_grid(z_slice_grids, rows=1)
            if draw_legend_z:
                z_texts = [[images.GridAnnotation(z)] for z in z_labels] if draw_legend_z else [[images.GridAnnotation()] for _ in z_labels]
                grid_max_w, grid_max_h = map(max, zip(*(img.size for img in z_slice_grids)))
                z_grid = images.draw_grid_annotations(z_grid, grid_max_w, grid_max_h, z_texts, [[images.GridAnnotation()]])
            final_grids.append(z_grid)
        else:
            grid = images.image_grid(batch_images, rows=len(ys))
            if draw_legend_x or draw_legend_y:
                hor_texts = [[images.GridAnnotation(x)] for x in x_labels] if draw_legend_x else [[images.GridAnnotation()] for _ in x_labels]
                ver_texts = [[images.GridAnnotation(y)] for y in y_labels] if draw_legend_y else [[images.GridAnnotation()] for _ in y_labels]
                grid_max_w, grid_max_h = map(max, zip(*(img.size for img in batch_images)))
                grid = images.draw_grid_annotations(grid, grid_max_w, grid_max_h, hor_texts, ver_texts, margin_size)
            final_grids.append(grid)

        final_processed = processed_template
        final_processed.images = final_grids + all_cell_images
        
        num_grids = len(final_grids)
        fallback_info = all_cell_infotexts[0] if all_cell_infotexts else ""
        fallback_prompt = all_cell_prompts[0] if all_cell_prompts else ""
        fallback_seed = all_cell_seeds[0] if all_cell_seeds else -1

        final_processed.infotexts = ([fallback_info] * num_grids) + all_cell_infotexts
        final_processed.all_prompts = ([fallback_prompt] * num_grids) + all_cell_prompts
        final_processed.all_seeds = ([fallback_seed] * num_grids) + all_cell_seeds
        final_processed.all_subseeds = ([-1] * num_grids) + all_cell_subseeds
        final_processed.all_negative_prompts = ([""] * num_grids) + all_cell_negative_prompts
        
        final_processed.index_of_first_image = num_grids

        return final_processed

def apply_xyz_patch(original_xyz_grid_module, original_script_class):
    """
    オリジナルのモジュールとクラスを受け取り、
    パッチ適用済みのUIおよびRUN関数を定義して差し替える
    """

    def patched_ui(self, is_img2img):
        """
        パッチを適用する UI 関数。
        """
        if not original_xyz_grid_module:
            return []
        
        self.current_axis_options = [x for x in original_xyz_grid_module.axis_options if type(x) == original_xyz_grid_module.AxisOption or getattr(x, 'is_img2img', False) == is_img2img]

        with gr.Row():
            with gr.Column(scale=19):
                with gr.Row():
                    x_type = gr.Dropdown(label="X type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[1].label, type="index", elem_id=self.elem_id("x_type"))
                    x_values = gr.Textbox(label="X values", lines=1, elem_id=self.elem_id("x_values"))
                    x_values_dropdown = gr.Dropdown(label="X values", visible=False, multiselect=True, interactive=True)
                    fill_x_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_x_tool_button", visible=False)
                with gr.Row():
                    y_type = gr.Dropdown(label="Y type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("y_type"))
                    y_values = gr.Textbox(label="Y values", lines=1, elem_id=self.elem_id("y_values"))
                    y_values_dropdown = gr.Dropdown(label="Y values", visible=False, multiselect=True, interactive=True)
                    fill_y_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_y_tool_button", visible=False)
                with gr.Row():
                    z_type = gr.Dropdown(label="Z type", choices=[x.label for x in self.current_axis_options], value=self.current_axis_options[0].label, type="index", elem_id=self.elem_id("z_type"))
                    z_values = gr.Textbox(label="Z values", lines=1, elem_id=self.elem_id("z_values"))
                    z_values_dropdown = gr.Dropdown(label="Z values", visible=False, multiselect=True, interactive=True)
                    fill_z_button = ToolButton(value=fill_values_symbol, elem_id="xyz_grid_fill_z_tool_button", visible=False)

        with gr.Row(variant="compact", elem_id="axis_options"):
            with gr.Column():
                no_fixed_seeds = gr.Checkbox(label='Keep -1 for seeds', value=False, elem_id=self.elem_id("no_fixed_seeds"))
                with gr.Row():
                    vary_seeds_x = gr.Checkbox(label='Vary seeds for X', value=False, min_width=80, elem_id=self.elem_id("vary_seeds_x"), tooltip="Use different seeds for images along X axis.")
                    vary_seeds_y = gr.Checkbox(label='Vary seeds for Y', value=False, min_width=80, elem_id=self.elem_id("vary_seeds_y"), tooltip="Use different seeds for images along Y axis.")
                    vary_seeds_z = gr.Checkbox(label='Vary seeds for Z', value=False, min_width=80, elem_id=self.elem_id("vary_seeds_z"), tooltip="Use different seeds for images along Z axis.")
            with gr.Column():
                include_lone_images = gr.Checkbox(label='Include Sub Images', value=False, elem_id=self.elem_id("include_lone_images"))
                csv_mode = gr.Checkbox(label='Use text inputs instead of dropdowns', value=False, elem_id=self.elem_id("csv_mode"))
                split_grids_by_batch = gr.Checkbox(label='Split grids by batch size', value=False, elem_id=self.elem_id("split_grids_by_batch"), tooltip="Instead of one grid, create N grids for batch size N.")

        with InputAccordion(True, label='Draw grid', elem_id=self.elem_id('draw_grid')) as draw_grid:
            with gr.Row():
                include_sub_grids = gr.Checkbox(label='Include Sub Grids', value=False, elem_id=self.elem_id("include_sub_grids"))
                margin_size = gr.Slider(label="Grid margins (px)", minimum=0, maximum=500, value=0, step=2, elem_id=self.elem_id("margin_size"))
            with gr.Row(elem_id="legend_options"):
                draw_legend_x = gr.Checkbox(label='Legend for X', value=True, elem_id=self.elem_id("draw_legend_x"))
                draw_legend_y = gr.Checkbox(label='Legend for Y', value=True, elem_id=self.elem_id("draw_legend_y"))
                draw_legend_z = gr.Checkbox(label='Legend for Z', value=True, elem_id=self.elem_id("draw_legend_z"))
                draw_legend_batch = gr.Checkbox(label='Legend for Batch Count', value=True, elem_id=self.elem_id("draw_legend_batch"))

        with gr.Row(variant="compact", elem_id="swap_axes"):
            swap_xy_axes_button = gr.Button(value="Swap X/Y axes", elem_id="xy_grid_swap_axes_button")
            swap_yz_axes_button = gr.Button(value="Swap Y/Z axes", elem_id="yz_grid_swap_axes_button")
            swap_xz_axes_button = gr.Button(value="Swap X/Z axes", elem_id="xz_grid_swap_axes_button")
        
        def swap_axes(axis1_type, axis1_values, axis1_values_dropdown, axis2_type, axis2_values, axis2_values_dropdown):
            return self.current_axis_options[axis2_type].label, axis2_values, axis2_values_dropdown, self.current_axis_options[axis1_type].label, axis1_values, axis1_values_dropdown

        xy_swap_args = [x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown]
        swap_xy_axes_button.click(swap_axes, inputs=xy_swap_args, outputs=xy_swap_args)
        yz_swap_args = [y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown]
        swap_yz_axes_button.click(swap_axes, inputs=yz_swap_args, outputs=yz_swap_args)
        xz_swap_args = [x_type, x_values, x_values_dropdown, z_type, z_values, z_values_dropdown]
        swap_xz_axes_button.click(swap_axes, inputs=xz_swap_args, outputs=xz_swap_args)

        def fill(axis_type, csv_mode):
            axis = self.current_axis_options[axis_type]
            if axis.choices:
                if csv_mode: return original_xyz_grid_module.list_to_csv_string(axis.choices()), gr.update()
                else: return gr.update(), axis.choices()
            else: return gr.update(), gr.update()

        fill_x_button.click(fn=fill, inputs=[x_type, csv_mode], outputs=[x_values, x_values_dropdown])
        fill_y_button.click(fn=fill, inputs=[y_type, csv_mode], outputs=[y_values, y_values_dropdown])
        fill_z_button.click(fn=fill, inputs=[z_type, csv_mode], outputs=[z_values, z_values_dropdown])

        def select_axis(axis_type, axis_values, axis_values_dropdown, csv_mode):
            axis_type = axis_type or 0
            choices = self.current_axis_options[axis_type].choices
            has_choices = choices is not None
            if has_choices:
                choices = choices()
                if csv_mode:
                    if axis_values_dropdown:
                        axis_values = original_xyz_grid_module.list_to_csv_string(list(filter(lambda x: x in choices, axis_values_dropdown)))
                        axis_values_dropdown = []
                else:
                    if axis_values:
                        axis_values_dropdown = list(filter(lambda x: x in choices, original_xyz_grid_module.csv_string_to_list_strip(axis_values)))
                        axis_values = ""
            return (gr.Button.update(visible=has_choices), gr.Textbox.update(visible=not has_choices or csv_mode, value=axis_values),
                    gr.update(choices=choices if has_choices else None, visible=has_choices and not csv_mode, value=axis_values_dropdown))

        x_type.change(fn=select_axis, inputs=[x_type, x_values, x_values_dropdown, csv_mode], outputs=[fill_x_button, x_values, x_values_dropdown])
        y_type.change(fn=select_axis, inputs=[y_type, y_values, y_values_dropdown, csv_mode], outputs=[fill_y_button, y_values, y_values_dropdown])
        z_type.change(fn=select_axis, inputs=[z_type, z_values, z_values_dropdown, csv_mode], outputs=[fill_z_button, z_values, z_values_dropdown])

        def change_choice_mode(csv_mode, x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown):
            _f_x, _x_v, _x_vd = select_axis(x_type, x_values, x_values_dropdown, csv_mode)
            _f_y, _y_v, _y_vd = select_axis(y_type, y_values, y_values_dropdown, csv_mode)
            _f_z, _z_v, _z_vd = select_axis(z_type, z_values, z_values_dropdown, csv_mode)
            return _f_x, _x_v, _x_vd, _f_y, _y_v, _y_vd, _f_z, _z_v, _z_vd

        csv_mode.change(fn=change_choice_mode, inputs=[csv_mode, x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown], outputs=[fill_x_button, x_values, x_values_dropdown, fill_y_button, y_values, y_values_dropdown, fill_z_button, z_values, z_values_dropdown])

        def get_dropdown_update_from_params(axis, params):
            val_key = f"{axis} Values"; vals = params.get(val_key, ""); valslist = original_xyz_grid_module.csv_string_to_list_strip(vals)
            return gr.update(value=valslist)

        self.infotext_fields = (
            (x_type, "X Type"), (x_values, "X Values"), (x_values_dropdown, lambda params: get_dropdown_update_from_params("X", params)),
            (y_type, "Y Type"), (y_values, "Y Values"), (y_values_dropdown, lambda params: get_dropdown_update_from_params("Y", params)),
            (z_type, "Z Type"), (z_values, "Z Values"), (z_values_dropdown, lambda params: get_dropdown_update_from_params("Z", params)),
        )
        
        return [x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown, include_lone_images, include_sub_grids, no_fixed_seeds, vary_seeds_x, vary_seeds_y, vary_seeds_z, margin_size, csv_mode, draw_grid, split_grids_by_batch, draw_legend_x, draw_legend_y, draw_legend_z, draw_legend_batch]

    def patched_run(self, p, x_type, x_values, x_values_dropdown, y_type, y_values, y_values_dropdown, z_type, z_values, z_values_dropdown, include_lone_images, include_sub_grids, no_fixed_seeds, vary_seeds_x, vary_seeds_y, vary_seeds_z, margin_size, csv_mode, draw_grid, split_grids_by_batch, draw_legend_x, draw_legend_y, draw_legend_z, draw_legend_batch):
        """
        パッチを適用する新しい run 関数。
        """
        if not original_xyz_grid_module:
            return Processed(p, [])
        
        x_type, y_type, z_type = x_type or 0, y_type or 0, z_type or 0
        if not no_fixed_seeds: processing.fix_seed(p)
        if not opts.return_grid: p.batch_size = 1
        
        list_to_csv_string = original_xyz_grid_module.list_to_csv_string
        csv_string_to_list_strip = original_xyz_grid_module.csv_string_to_list_strip
        re_range = original_xyz_grid_module.re_range
        re_range_count = original_xyz_grid_module.re_range_count
        re_range_float = original_xyz_grid_module.re_range_float
        re_range_count_float = original_xyz_grid_module.re_range_count_float
        SharedSettingsStackHelper = original_xyz_grid_module.SharedSettingsStackHelper
        AxisInfo = original_xyz_grid_module.AxisInfo
        str_permutations = original_xyz_grid_module.str_permutations

        def process_axis(opt, vals, vals_dropdown):
            if opt.label == 'Nothing': return [0]
            if opt.choices is not None and not csv_mode: valslist = vals_dropdown
            elif opt.prepare is not None: valslist = opt.prepare(vals)
            else: valslist = csv_string_to_list_strip(vals)

            if opt.type == int:
                valslist_ext = []
                for val in valslist:
                    if val.strip() == '': continue
                    m = re_range.fullmatch(val); mc = re_range_count.fullmatch(val)
                    if m is not None: start = int(m.group(1)); end = int(m.group(2)) + 1; step = int(m.group(3)) if m.group(3) is not None else 1; valslist_ext += list(range(start, end, step))
                    elif mc is not None: start = int(mc.group(1)); end = int(mc.group(2)); num = int(mc.group(3)) if mc.group(3) is not None else 1; valslist_ext += [int(x) for x in np.linspace(start=start, stop=end, num=num).tolist()]
                    else: valslist_ext.append(val)
                valslist = valslist_ext
            elif opt.type == float:
                valslist_ext = []
                for val in valslist:
                    if val.strip() == '': continue
                    m = re_range_float.fullmatch(val); mc = re_range_count_float.fullmatch(val)
                    if m is not None: start = float(m.group(1)); end = float(m.group(2)); step = float(m.group(3)) if m.group(3) is not None else 1; valslist_ext += np.arange(start, end + step, step).tolist()
                    elif mc is not None: start = float(mc.group(1)); end = float(mc.group(2)); num = int(mc.group(3)) if mc.group(3) is not None else 1; valslist_ext += np.linspace(start=start, stop=end, num=num).tolist()
                    else: valslist_ext.append(val)
                valslist = valslist_ext
            elif opt.type == str_permutations:
                valslist = list(permutations(valslist))

            valslist = [opt.type(x) for x in valslist]
            if opt.confirm: opt.confirm(p, valslist)
            return valslist

        x_opt = self.current_axis_options[x_type]
        if x_opt.choices is not None and not csv_mode: x_values = list_to_csv_string(x_values_dropdown)
        xs = process_axis(x_opt, x_values, x_values_dropdown)
        y_opt = self.current_axis_options[y_type]
        if y_opt.choices is not None and not csv_mode: y_values = list_to_csv_string(y_values_dropdown)
        ys = process_axis(y_opt, y_values, y_values_dropdown)
        z_opt = self.current_axis_options[z_type]
        if z_opt.choices is not None and not csv_mode: z_values = list_to_csv_string(z_values_dropdown)
        zs = process_axis(z_opt, z_values, z_values_dropdown)

        Image.MAX_IMAGE_PIXELS = None
        grid_mp = round(len(xs) * len(ys) * len(zs) * p.width * p.height / 1000000)
        assert grid_mp < opts.img_max_size_mp, f'Error: Resulting grid would be too large ({grid_mp} MPixels)'

        def fix_axis_seeds(axis_opt, axis_list):
            if axis_opt.label in ['Seed', 'Var. seed']: return [int(random.randrange(4294967294)) if val is None or val == '' or val == -1 else val for val in axis_list]
            else: return axis_list
        if not no_fixed_seeds:
            xs = fix_axis_seeds(x_opt, xs); ys = fix_axis_seeds(y_opt, ys); zs = fix_axis_seeds(z_opt, zs)

        image_cell_count = p.n_iter * p.batch_size
        state.xyz_plot_x = AxisInfo(x_opt, xs); state.xyz_plot_y = AxisInfo(y_opt, ys); state.xyz_plot_z = AxisInfo(z_opt, zs)

        grid_infotext = [None] * (1 + len(zs))

        def cell(x, y, z, ix, iy, iz):
            if shared.state.interrupted or state.stopping_generation: return Processed(p, [], p.seed, "")
            pc = copy(p)
            
            # --- Forge/SDXL 参照共有回避 ---
            pc.cached_c = [None, None]
            pc.cached_uc = [None, None]
            if hasattr(pc, 'cached_hr_c'):  pc.cached_hr_c = [None, None]
            if hasattr(pc, 'cached_hr_uc'): pc.cached_hr_uc = [None, None]
            pc.extra_network_data = None
            if p.all_prompts: pc.all_prompts = list(p.all_prompts)
            else: pc.all_prompts = [p.prompt] * p.batch_size
            if p.all_negative_prompts: pc.all_negative_prompts = list(p.all_negative_prompts)
            else: pc.all_negative_prompts = [p.negative_prompt] * p.batch_size
            if p.all_seeds: pc.all_seeds = list(p.all_seeds)
            else: pc.all_seeds = [p.seed] * p.batch_size
            if p.all_subseeds: pc.all_subseeds = list(p.all_subseeds)
            else: pc.all_subseeds = [p.subseed] * p.batch_size
            # ---------------------------

            pc.styles = pc.styles[:]
            x_opt.apply(pc, x, xs); y_opt.apply(pc, y, ys); z_opt.apply(pc, z, zs)
            xdim = len(xs) if vary_seeds_x else 1; ydim = len(ys) if vary_seeds_y else 1
            if vary_seeds_x: pc.seed += ix
            if vary_seeds_y: pc.seed += iy * xdim
            if vary_seeds_z: pc.seed += iz * xdim * ydim
            
            try: res = process_images(pc)
            except Exception as e: errors.display(e, "generating image for xyz plot"); res = Processed(p, [], p.seed, "")
            
            subgrid_index = 1 + iz
            if grid_infotext[subgrid_index] is None and ix == 0 and iy == 0:
                pc.extra_generation_params = copy(pc.extra_generation_params); pc.extra_generation_params['Script'] = self.title()
                if x_opt.label != 'Nothing': pc.extra_generation_params["X Type"] = x_opt.label; pc.extra_generation_params["X Values"] = x_values
                if y_opt.label != 'Nothing': pc.extra_generation_params["Y Type"] = y_opt.label; pc.extra_generation_params["Y Values"] = y_values
                grid_infotext[subgrid_index] = processing.create_infotext(pc, pc.all_prompts, pc.all_seeds, pc.all_subseeds)
            if grid_infotext[0] is None and ix == 0 and iy == 0 and iz == 0:
                pc.extra_generation_params = copy(pc.extra_generation_params)
                if z_opt.label != 'Nothing': pc.extra_generation_params["Z Type"] = z_opt.label; pc.extra_generation_params["Z Values"] = z_values
                grid_infotext[0] = processing.create_infotext(pc, pc.all_prompts, pc.all_seeds, pc.all_subseeds)
            return res

        with SharedSettingsStackHelper():
            if split_grids_by_batch and p.batch_size > 1:
                p.override_settings['grid_save'] = False
            
            processed = draw_xyz_grid_unified(
                p, xs=xs, ys=ys, zs=zs, 
                x_labels=[x_opt.format_value(p, x_opt, x) for x in xs], 
                y_labels=[y_opt.format_value(p, y_opt, y) for y in ys], 
                z_labels=[z_opt.format_value(p, z_opt, z) for z in zs], 
                cell=cell, margin_size=margin_size, 
                draw_legend_x=draw_legend_x, draw_legend_y=draw_legend_y, draw_legend_z=draw_legend_z, draw_legend_batch=draw_legend_batch,
                include_sub_grids=include_sub_grids, split_grids_by_batch=split_grids_by_batch
            )
            
            # Forge 固有のリフレッシュ処理
            if hasattr(original_xyz_grid_module, 'refresh_loading_params_for_xyz_grid'):
                original_xyz_grid_module.refresh_loading_params_for_xyz_grid()

            p.override_settings.pop('grid_save', None)
            if not processed.images: return processed
            
            # メイン/サブグリッドのメタデータ上書き
            if draw_grid:
                 num_infotexts_to_copy = min(len(processed.infotexts), 1 + len(zs))
                 processed.infotexts[:num_infotexts_to_copy] = grid_infotext[:num_infotexts_to_copy]

            # 個別画像（Include Sub Images）が不要な場合のフィルタリング
            if not include_lone_images:
                 num_grids_to_keep = processed.index_of_first_image
                 processed.images = processed.images[:num_grids_to_keep]
                 processed.infotexts = processed.infotexts[:num_grids_to_keep]
                 
                 # メタデータも全てスライスして同期させる
                 processed.all_prompts = processed.all_prompts[:num_grids_to_keep]
                 processed.all_seeds = processed.all_seeds[:num_grids_to_keep]
                 if hasattr(processed, 'all_subseeds'):
                     processed.all_subseeds = processed.all_subseeds[:num_grids_to_keep]
                 if hasattr(processed, 'all_negative_prompts'):
                     processed.all_negative_prompts = processed.all_negative_prompts[:num_grids_to_keep]

            # グリッド保存
            if opts.grid_save:
                for i in range(processed.index_of_first_image):
                    images.save_image(processed.images[i], p.outpath_grids, "xyz_grid", info=processed.infotexts[i], extension=opts.grid_format, prompt=processed.all_prompts[i], seed=processed.all_seeds[i], grid=True, p=p)

            return processed
    
    # --- パッチ適用の本体 ---
    print("[XYZ Plot Mods] Applying patches...")
    original_xyz_grid_module.draw_xyz_grid_unified = draw_xyz_grid_unified
    original_script_class.ui = patched_ui
    original_script_class.run = patched_run
    print("[XYZ Plot Mods] Patches applied successfully.")

def on_scripts_loaded():
    """
    すべてのスクリプトが読み込まれた後に呼び出されるコールバック関数。
    """
    print("[XYZ Plot Mods] Initializing via callback...")
    try:
        target_module_names = {"xyz_grid.py", "xy_grid.py", "scripts.xyz_grid", "scripts.xy_grid"}
        original_script_class = None
        original_xyz_grid_module = None

        if hasattr(scripts, "scripts_data") and isinstance(scripts.scripts_data, list):
            for data in scripts.scripts_data:
                if not hasattr(data, "script_class"): continue
                module_name = data.script_class.__module__
                if module_name in target_module_names:
                    try:
                        original_script_class = data.script_class
                        original_xyz_grid_module = data.module if hasattr(data, "module") else importlib.import_module(module_name)
                        print(f"[XYZ Plot Mods] Found original script! Class: {original_script_class}")
                        break 
                    except Exception as e:
                        print(f"[XYZ Plot Mods] Error importing module: {e}")
                        
        if original_script_class and original_xyz_grid_module:
            apply_xyz_patch(original_xyz_grid_module, original_script_class)
        else:
            print(f"[XYZ Plot Mods] Error: Could not find xyz_grid script.")

    except Exception as e:
        print(f"[XYZ Plot Mods] Critical error: {e}")
        traceback.print_exc()

# --- コールバックを登録 ---
script_callbacks.on_ui_settings(on_scripts_loaded)
