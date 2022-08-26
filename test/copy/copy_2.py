#!/usr/bin/env python
# coding=utf-8
import os
import multiprocessing


def copy_file(q, file_name, old_folder_name, new_folder_name):
    """完成文件复制"""
    #  print("=====>模拟copy文件：从%s----->到%s 文件名是：%s" % (old_folder_name, new_folder_name, file_name))
    old_f = open(old_folder_name + "/" + file_name, "rb")
    content = old_f.read()
    old_f.close()

    new_f = open(new_folder_name + "/" + file_name, "wb")
    new_f.write(content)
    new_f.close()

    # 如果copy结束，就向队列中写入一个消息，表示已经完成
    q.put(file_name)


def main():
    # 1. 获取用户要copy文件夹的名字
    # old_folder_name = input("please input copy_folder_name:")
    old_folder_name = r'F:\余额'
    # 2. 创建一个新的文件夹
    try:
        new_folder_name = old_folder_name + "[复件]"
        os.mkdir(old_folder_name + "[复件]")
    except:
        pass
    # 3. 获取文件夹中所有带copy文件的名字 listdir()
    file_names = os.listdir(old_folder_name)
    # print(file_names)

    # 4. 创建进程池
    po = multiprocessing.Pool(5)

    # 5. 创建一个队列
    q = multiprocessing.Manager().Queue()

    # 6. 向进程池中添加copy文件的任务
    for file_name in file_names:
        po.apply_async(copy_file, args=(q, file_name, old_folder_name, new_folder_name))

    #  复制源文件夹中的文件到新文件夹中的文件去

    po.close()
    # po.join()
    # 测一下所有文件个数
    all_file_num = len(file_names)
    copy_complete_num = 0
    while True:
        file_name = q.get()
        # print("已经完成copy：%s" % file_name)

        copy_complete_num += 1
        print("\r拷贝进度为: %.2f %%" % (copy_complete_num * 100 / all_file_num), end="")
        if copy_complete_num >= all_file_num:
            break

    print()


if __name__ == "__main__":
    main()

