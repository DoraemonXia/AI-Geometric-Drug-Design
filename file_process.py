import os
from PIL import Image
from reportlab.pdfgen import canvas

#process png to pdf
def images_to_pdf(folder_path, output_pdf, include_title=False):
    # 获取所有的png文件，并按名称排序
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])

    # 创建一个PDF文件
    c = canvas.Canvas(output_pdf)

    for image_file in image_files:
        img_path = os.path.join(folder_path, image_file)
        img = Image.open(img_path)

        # 获取图像尺寸，并转换为PDF的单位（pt）
        width, height = img.size
        width, height = width * 0.75, height * 0.75  # assuming 1 pixel = 0.75 pt

        # 设定页尺寸
        if include_title:
            c.setPageSize((width, height + 50))  # 增加空间用于标题
            c.drawString(10, height + 20, image_file)  # 在顶部绘制标题
        else:
            c.setPageSize((width, height))

        # 绘制图像
        c.drawImage(img_path, 0, 0, width=width, height=height)
        c.showPage()  # 新的一页

    c.save()

# folder_path = "source_data/Robin/Robin_png/"  # 替换为你的文件夹路径
# output_pdf = "Robin_RNA.pdf"  # 输出PDF文件的路径
# include_title = True  # 是否在每页添加标题
# images_to_pdf(folder_path, output_pdf, include_title)