import modules.scripts as scripts
import gradio as gr

class SeedStepCustomizer(scripts.Script):
    def title(self):
        return "Custom Seed Step Incrementor"

    def show(self, is_img2img):
        # ALWAYS 表示にして常時パネル下部（UI）に表示させる
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Seed Step Customizer", open=False):
            enabled = gr.Checkbox(label="Enable Custom Seed Step", value=False)
            step_size = gr.Number(label="Seed Step Size", value=10, precision=0)
        return [enabled, step_size]

    def process(self, p, enabled, step_size):
        if not enabled or step_size is None or step_size == 1:
            return

        step = int(step_size)
        
        # p.all_seeds に全バッチ・全サブバッチ分のSeedが割り振られている
        if hasattr(p, "all_seeds") and p.all_seeds:
            base_seed = p.all_seeds[0]
            # 指定ステップ数でSeed配列を再構築
            # 例: Step=10, Base=1000 -> [1000, 1010, 1020, 1030, ...]
            p.all_seeds = [base_seed + (i * step) for i in range(len(p.all_seeds))]
            
            # 主Seed（p.seed）も最新状態に整合させる
            p.seed = p.all_seeds[0]
