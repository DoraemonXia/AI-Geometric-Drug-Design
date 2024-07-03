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


from PIL import Image
#need pip install pillow

def convert_white_to_transparent(input_image_path, output_image_path):
    """
    Transfer graph with white color as background color into opticify as background color.
    """
    image = Image.open(input_image_path).convert("RGBA")

    # 获取图像的像素数据
    datas = image.getdata()

    new_data = []
    for item in datas:
        # 改变白色（也是透明的）背景为透明
        if item[0] > 200 and item[1] > 200 and item[2] > 200:
            new_data.append((255, 255, 255, 0))  # 更改透明
        else:
            new_data.append(item)

    # 更新图像的数据
    image.putdata(new_data)

    # 保存图像
    image.save(output_image_path, "PNG")


"""
Example Usage:
input_image_path = "pic/Pic_new/SAM_II.png"  # input filepath of graph
output_image_path = "pic/Pic_new/SAM_II_new.png"  # output filepath of graph

convert_white_to_transparent(input_image_path, output_image_path)
print(f"Saved transparent image to {output_image_path}")
"""
