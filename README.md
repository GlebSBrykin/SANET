# SANET

This is unofficial PyTorch implementation of "Arbitrary Style Transfer with Style-Attentional Networks".

Official paper: https://arxiv.org/abs/1812.02342v5

To run, download the weights and place them in the folder with Eval.py. Links to weights on Yandex.Disk:

* decoder: https://yadi.sk/d/xsZ7j6FhK1dmfQ

* transformer: https://yadi.sk/d/GhQe3g_iRzLKMQ

* vgg_normalised: https://yadi.sk/d/7IrysY8q8dtneQ

Or, you can download the latest release. It contains all weights, codes and examples.

# How to evaluate

To test the code, make changes to the following lines in the file Eval.py. here you need to specify the path to the image style and content. After that, save the changes to the file and run it.

```python
parser.add_argument('--content', type=str, default = 'input/chicago.jpg',
                    help='File path to the content image')
parser.add_argument('--style', type=str, default = 'style/style11.jpg',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
```

# Examples

Original:

![Content.jpg](https://github.com/GlebBrykin/SANET/blob/master/input/Content.jpg)

Stylized under 1.jpg:

![Content_stylized_1.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/Content_stylized_1.jpg)

Stylized under Composition-VII.jpg:

![Content_stylized_Composition-VII.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/Content_stylized_Composition-VII.jpg)

Stylized under Starry.jpg:

![Content_stylized_Starry.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/Content_stylized_Starry.jpg)

Stylized under candy.jpg:

![Content_stylized_candy.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/Content_stylized_candy.jpg)

Stylized under la_muse.jpg:

![Content_stylized_la_muse.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/Content_stylized_la_muse.jpg)

Stylized under rain_princess.jpg:

![Content_stylized_rain_princess.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/Content_stylized_rain_princess.jpg)

Stylized under seated_nude.jpg:

![Content_stylized_seated_nude.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/Content_stylized_seated_nude.jpg)

Stylized under style11.jpg:

![Content_stylized_style11.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/Content_stylized_style11.jpg)

Stylized under udnie.jpg:

![Content_stylized_udnie.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/Content_stylized_udnie.jpg)

Stylized under wave.jpg:

![Content_stylized_wave.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/Content_stylized_wave.jpg)

Stylized under wreck.jpg:

![Content_stylized_wreck.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/Content_stylized_wreck.jpg)

Original:

![chicago.jpg](https://github.com/GlebBrykin/SANET/blob/master/input/chicago.jpg)

Stylized under 1.jpg:

![chicago_stylized_1.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/chicago_stylized_1.jpg)

Stylized under Composition-VII.jpg:

![chicago_stylized_Composition-VII.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/chicago_stylized_Composition-VII.jpg)

Stylized under Starry.jpg:

![chicago_stylized_Starry.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/chicago_stylized_Starry.jpg)

Stylized under candy.jpg:

![chicago_stylized_candy.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/chicago_stylized_candy.jpg)

Stylized under la_muse.jpg:

![chicago_stylized_la_muse.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/chicago_stylized_la_muse.jpg)

Stylized under rain_princess.jpg:

![chicago_stylized_rain_princess.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/chicago_stylized_rain_princess.jpg)

Stylized under seated_nude.jpg:

![chicago_stylized_seated_nude.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/chicago_stylized_seated_nude.jpg)

Stylized under style11.jpg:

![chicago_stylized_style11.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/chicago_stylized_style11.jpg)

Stylized under udnie.jpg:

![chicago_stylized_udnie.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/chicago_stylized_udnie.jpg)

Stylized under wave.jpg:

![chicago_stylized_wave.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/chicago_stylized_wave.jpg)

Stylized under wreck.jpg:

![chicago_stylized_wreck.jpg](https://github.com/GlebBrykin/SANET/blob/master/output/chicago_stylized_wreck.jpg)
