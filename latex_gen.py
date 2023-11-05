import glob
import os

path_ufo = '/home/kaien125/experiments/paper/ufo_yolo_compare/ufo'
path_yolo = '/home/kaien125/experiments/paper/ufo_yolo_compare/yolo'

ufo_ids = sorted(glob.glob(os.path.join(path_ufo, '*.jpg')))
yolo_ids = sorted(glob.glob(os.path.join(path_yolo, '*.jpg')))

line_begin = r'\begin{figure}'
line_center = r'\centering'
line_begin_sub = r'\begin{subfigure}{.5\textwidth}'
line_include = r'\includegraphics[width=1\linewidth]'
line_end_sub = r'\end{subfigure}%'
line_end = r'\end{figure}'

line1 = r'\RequirePackage[2020-02-02]{latexrelease}'
line2 = r'\documentclass[11pt]{article}'
line3 = r'\usepackage[export]{adjustbox}'
line4 = r'\usepackage{subcaption}'
line5 = r'\begin{document}'
line6 = r'\end{document}'

with open('/home/kaien125/experiments/paper/ufo_yolo_compare/compare_latex.txt', "w") as file:
    # file.write('UFOnet outputs are green bbox on the left. YOLOv4 tiny outputs are red bbox on the right.')
    file.write("%s\n %s\n %s\n %s\n %s\n"
               % (line1, line2, line3, line4, line5))

    for ufo_id, yolo_id in zip(ufo_ids, yolo_ids):
        line_ufo = '{' + ufo_id + '}'
        line_yolo = '{' + yolo_id + '}'

        file.write("%s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n"
                   % (line_begin, line_center, line_begin_sub, line_center,
                      line_include, line_ufo, line_end_sub, line_begin_sub,
                      line_center, line_include, line_yolo, line_end_sub, line_end))

    file.write(line6)







