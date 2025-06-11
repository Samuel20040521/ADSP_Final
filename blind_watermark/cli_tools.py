from optparse import OptionParser
from .blind_watermark import WaterMark
from . import att
import cv2
import numpy as np
import os

usage1 = 'blind_watermark --embed --pwd 1234 image.jpg "watermark text" embed.png'
usage2 = 'blind_watermark --extract --pwd 1234 --wm_shape 111 embed.png'
usage3 = 'blind_watermark --attack shelter embed.png attacked.png'
usage4 = 'blind_watermark --attack shelter embed.png attacked.png --origin ori.jpg --wm_text "wm" --compare compare.png'
optParser = OptionParser(usage='\n'.join([usage1, usage2, usage3, usage4]))

optParser.add_option('--embed', dest='work_mode', action='store_const', const='embed'
                     , help='Embed watermark into images')
optParser.add_option('--extract', dest='work_mode', action='store_const', const='extract'
                     , help='Extract watermark from images')

optParser.add_option('-p', '--pwd', dest='password', help='password, like 1234')
optParser.add_option('--wm_shape', dest='wm_shape', help='Watermark shape, like 120')
optParser.add_option('--attack', dest='attack', help='attack mode, like shelter')
optParser.add_option('--compare', dest='compare', help='path to save comparison image')
optParser.add_option('--origin', dest='origin', help='original image for comparison')
optParser.add_option('--wm_text', dest='wm_text', help='watermark text for comparison')
optParser.add_option('--wm_img', dest='wm_img', help='watermark image file for comparison')

(opts, args) = optParser.parse_args()


def _text_to_img(text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, 1, 2)
    img = np.ones((h + baseline + 20, w + 20, 3), dtype=np.uint8) * 255
    cv2.putText(img, text, (10, h + baseline + 5), font, 1, (0, 0, 0), 2)
    return img


def _load_wm(opts, wm_arg=None):
    if opts.wm_img:
        return cv2.imread(opts.wm_img)
    text = opts.wm_text if opts.wm_text is not None else wm_arg
    if text is None:
        return None
    return _text_to_img(text)


def make_compare(wm_arg, processed_file, origin_file, out_file):
    wm_img = _load_wm(opts, wm_arg)
    if wm_img is None:
        print('Watermark for comparison not specified')
        return
    img_p = cv2.imread(processed_file)
    img_o = cv2.imread(origin_file)
    if img_p is None or img_o is None:
        print('Images for comparison not found')
        return
    h = max(wm_img.shape[0], img_p.shape[0], img_o.shape[0])
    def resize(img):
        return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
    comp = cv2.hconcat([resize(wm_img), resize(img_p), resize(img_o)])
    cv2.imwrite(out_file, comp)
    print('Comparison image saved to', out_file)


def main():
    bwm1 = WaterMark(password_img=int(opts.password))
    if opts.attack:
        attacks = {
            'shelter': att.shelter_att,
            'salt_pepper': att.salt_pepper_att,
            'resize': att.resize_att,
            'bright': att.bright_att,
            'rotate': att.rot_att,
            'cut': att.cut_att3,
        }
        if opts.attack not in attacks:
            print('Unknown attack: {}'.format(opts.attack))
            print(usage3)
            return
        if not len(args) == 2:
            print('Error! Usage: ')
            print(usage3)
            return
        attacks[opts.attack](input_filename=args[0], output_file_name=args[1])
        print('Attack {} succeed! to file {}'.format(opts.attack, args[1]))
        if opts.compare and opts.origin:
            make_compare(None, args[1], opts.origin, opts.compare)

    if opts.work_mode == 'embed':
        if not len(args) == 3:
            print('Error! Usage: ')
            print(usage1)
            return
        else:
            bwm1.read_img(args[0])
            bwm1.read_wm(args[1], mode='str')
            bwm1.embed(args[2])
            print('Embed succeed! to file ', args[2])
            print('Put down watermark size:', len(bwm1.wm_bit))
            if opts.compare:
                make_compare(args[1], args[2], args[0], opts.compare)

    if opts.work_mode == 'extract':
        if not len(args) == 1:
            print('Error! Usage: ')
            print(usage2)
            return

        else:
            wm_str = bwm1.extract(filename=args[0], wm_shape=int(opts.wm_shape), mode='str')
            print('Extract succeed! watermark is:')
            print(wm_str)


'''
python -m blind_watermark.cli_tools --embed --pwd 1234 examples/pic/ori_img.jpeg "watermark text" examples/output/embedded.png
python -m blind_watermark.cli_tools --extract --pwd 1234 --wm_shape 111 examples/output/embedded.png


cd examples
blind_watermark --embed --pwd 1234 examples/pic/ori_img.jpeg "watermark text" examples/output/embedded.png
blind_watermark --extract --pwd 1234 --wm_shape 111 examples/output/embedded.png
'''
