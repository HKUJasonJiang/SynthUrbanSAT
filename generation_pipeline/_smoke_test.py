"""End-to-end smoke test: load checkpoint, encode prompt, generate ours + both
baselines for 1 seed, save run. Prints PASS/FAIL.
"""
import os
import sys, json, time, traceback
sys.path.insert(0, '.')

from pipeline import list_lora_checkpoints
from pipeline.state import STATE
import app as A

GOLDEN = os.environ.get('SYNTHURBANSAT_GOLDEN_SET', '../train_pipeline/dataset/val')
STEM = 'JAX_Tile_115_002_1'
seg_p = f'{GOLDEN}/seg/{STEM}.png'
dep_p = f'{GOLDEN}/depth/{STEM}.tif'
rgb_p = f'{GOLDEN}/rgb/{STEM}.tif'

def main():
    ckpts = list_lora_checkpoints()
    assert ckpts, 'no checkpoints'
    print(f'> Loading {ckpts[0].name}')
    STATE.load(ckpts[0], persistent_text_encoder=True)

    print('> Encoding prompt')
    embed, flat, msg = A.encode_prompt_ui(A.DEFAULT_PROMPT_TEXT)
    A._LAST_PROMPT_JSON['text'] = A.DEFAULT_PROMPT_TEXT
    assert embed is not None, msg
    print('  ', msg[:160])

    print('> Generating (1 seed, 6 steps, with both baselines)')
    t0 = time.time()
    result = A.generate(
        seg_p, dep_p, rgb_p, embed,
        seeds_str='0', num_steps=6, guidance_scale=3.5,
        run_baseline_seg=True, run_baseline_depth=True,
    )
    seg_v, dep_v, feat_v, ours_v, base_seg_v, base_dep_v, summ_v, status, run_state = result
    print(f'  took {time.time()-t0:.1f}s -- status: {status[:160]}')
    assert ours_v is not None, 'ours_v is None'
    assert base_seg_v is not None, 'base_seg_v is None'
    assert base_dep_v is not None, 'base_dep_v is None'
    assert summ_v is not None, 'summ_v is None'
    print(f'  ours shape={ours_v.shape}  base_seg shape={base_seg_v.shape}  base_dep shape={base_dep_v.shape}  summary shape={summ_v.shape}')

    print('> Saving run')
    from pipeline.save import save_run
    run_dir, files = save_run(
        'smoke_test',
        seg_src_path=run_state.get('seg_src'),
        depth_src_path=run_state.get('depth_src'),
        rgb_src_path=run_state.get('rgb_src'),
        seg_preview=run_state.get('seg_rgb'),
        depth_preview=run_state.get('depth_rgb'),
        rgb_preview=run_state.get('gt_rgb'),
        feature_preview=run_state.get('feature_rgb'),
        ours_tiles=run_state.get('ours_tiles'),
        baseline_seg_tiles=run_state.get('baseline_seg_tiles'),
        baseline_depth_tiles=run_state.get('baseline_depth_tiles'),
        summary_grid=run_state.get('summary'),
        prompt_json_text=A.DEFAULT_PROMPT_TEXT,
        flat_prompt=run_state.get('flat_prompt', ''),
        seeds=run_state.get('seeds', []),
        metadata_extra={'smoke_test': True},
    )
    print(f'  saved {len(files)} files to {run_dir}')
    print('\nPASS')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        print(f'\nFAIL: {e}')
        sys.exit(1)
