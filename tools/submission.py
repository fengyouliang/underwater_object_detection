import os

from pathlib2 import Path
from tqdm import tqdm

avalible_gpu_ids = [0]
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(list(map(str, avalible_gpu_ids)))

from mmdet.apis import init_detector, inference_detector


def submission_test(config_file, checkpoint_file, save_path):
    model = init_detector(config_file, checkpoint_file)

    classes = ('holothurian', 'echinus', 'scallop', 'starfish')

    results = ['name,image_id,confidence,xmin,ymin,xmax,ymax\n']

    test_path = Path('/home/fengyouliang/datasets/underwater/test/test-A-image')
    all_test_images = list(test_path.rglob('*.jpg'))

    image_bar = tqdm(all_test_images)
    for image in image_bar:
        result = inference_detector(model, image.as_posix())
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None

        for class_idx, class_result in enumerate(bbox_result):
            for item in class_result:
                x1, y1, x2, y2, score = item
                csv_line = f"{classes[class_idx]},{image.stem},{score},{round(x1)},{round(y1)},{round(x2)},{round(y2)}\n"
                results.append(csv_line)

    with open(save_path, 'w', encoding='utf-8') as fid:
        fid.writelines(results)

    print(f'saved in {save_path}')


def main(model_type):
    project_prefix = '/home/fengyouliang/code/open_mmlab_codebase/project/underwater'
    config_file = f'{project_prefix}/work_dirs/{model_type}/{model_type}.py'
    checkpoint_file = f'{project_prefix}/work_dirs/{model_type}/latest.pth'
    submission_path = f'{project_prefix}/submission'

    submission_test(config_file, checkpoint_file, save_path=f"{submission_path}/{model_type}.csv")


if __name__ == '__main__':
    main('fcos')
