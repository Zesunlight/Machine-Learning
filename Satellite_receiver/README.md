 # 图片标注说明

## 标注工具

- labelImg
- https://github.com/Xminghua/labelImg

## 运行方式

- labelImg.exe
- Open Dir  选择图片所在的文件夹
- Change Save Dir  选择标签存储的位置
- **标签的存储格式选择 YOLO**（默认是 PscalVOC，点一下就变成 YOLO 了）
- Create RectBox  在图片上划出目标的框，选择相应标签
- **保存后**再标注下一个图

## 快捷键

| Hotkeys  | description                              |
| -------- | ---------------------------------------- |
| Ctrl + u | Load all of the images from a directory  |
| Ctrl + r | Change the default annotation target dir |
| Ctrl + s | Save                                     |
| Ctrl + d | Copy the current label and rect box      |
| w        | Create a rect box                        |
| d        | Next image                               |
| a        | Previous image                           |
| del      | Delete the selected rect box             |
| Ctrl++   | Zoom in                                  |
| Ctrl--   | Zoom out                                 |
| ↑→↓←     | Keyboard arrows to move selected rect box |

## 标注说明

- 密集的多个小目标可以用一个大框全部标注
- 肉眼清晰可见的单个目标用一个框标注
- 图片可以放大后再标注


