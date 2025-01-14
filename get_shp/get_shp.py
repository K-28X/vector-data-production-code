import ctypes
import sys
import os
import io
import threading

from tqdm import tqdm
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')  # 打印出中文字符
try:
    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr
except ImportError:
    import gdal
    import ogr
    import osr


def transfer_16_to_8(pInput, pOutput, DLLPath, DLLName):
    os.chdir(DLLPath)
    print(os.getcwd())
    SimDeepDll = ctypes.cdll.LoadLibrary(os.path.join(DLLPath, DLLName))
    # SimDeepDll = ctypes.CDLL(os.path.join(DLLPath, DLLName))
    SimDeepDll.gDosStrech16To8(pInput.encode(encoding='UTF-8'),
                               pOutput.encode(encoding='UTF-8'))

# def ExtractField(DLLPath, DLLName, pInput, pSS, pTempPath, pAThinPath, pRetPath, pRetGEOPath):
#     os.chdir(DLLPath)
#     print(os.getcwd())
#     print(os.path.join(DLLPath, DLLName))
#     # SimDeepDll = ctypes.cdll.LoadLibrary(os.path.join(DLLPath, DLLName))
#     SimDeepDll = ctypes.CDLL(os.path.join(DLLPath, DLLName))
#     SimDeepDll.ExtractFieldSSEdgeLargeFile(pInput.encode(encoding='UTF-8'),
#                                            pSS.encode(encoding='UTF-8'),
#                                            pTempPath.encode(encoding='UTF-8'),
#                                            pAThinPath.encode(encoding='UTF-8'),
#                                            pRetPath.encode(encoding='UTF-8'),
#                                            pRetGEOPath.encode(encoding='UTF-8'))

def ExtractField2(DLLPath, DLLName, pInput, pSS, pTempPath, pAThinPath, pRetPath, pRetGEOPath,pAThinRes):
    os.chdir(DLLPath)
    print(os.getcwd())
    print(os.path.join(DLLPath, DLLName))
    # SimDeepDll = ctypes.cdll.LoadLibrary(os.path.join(DLLPath, DLLName))
    SimDeepDll = ctypes.CDLL(os.path.join(DLLPath, DLLName))

    # SimDeepDll.DoNMSLargeFile(pInput.encode(encoding='UTF-8'),pAThinPath.encode(encoding='UTF-8'))
    SimDeepDll.ExtractFieldSSEdgeLargeFileNMS(pInput.encode(encoding='UTF-8'),
                                           pSS.encode(encoding='UTF-8'),
                                           pAThinPath.encode(encoding='UTF-8'),
                                           pTempPath.encode(encoding='UTF-8'),
                                           pAThinRes.encode(encoding='UTF-8'),
                                           pRetPath.encode(encoding='UTF-8'),
                                           pRetGEOPath.encode(encoding='UTF-8'),
                                           True)

def ExtractField1(DLLPath, DLLName, pInput, pSS, pTempPath, pAThinPath, pRetPath, pRetGEOPath,pAThinRes):
    os.chdir(DLLPath)
    print(os.getcwd())
    print(os.path.join(DLLPath, DLLName))
    SimDeepDll = ctypes.cdll.LoadLibrary(os.path.join(DLLPath, DLLName))
    SimDeepDll = ctypes.CDLL(os.path.join(DLLPath, DLLName))
    SimDeepDll.DoNMSLargeFile(pInput.encode(encoding='UTF-8'),pAThinPath.encode(encoding='UTF-8'))

# def ExtractField1(DLLPath, DLLName, pInput_diff, pSS, pTempPath, pAThinPath, pRetPath, pRetGEOPath,pAThinRes):
#     os.chdir(DLLPath)
#     print(os.getcwd())
#     print(os.path.join(DLLPath, DLLName))
#     SimDeepDll = ctypes.cdll.LoadLibrary(os.path.join(DLLPath, DLLName))
#     SimDeepDll = ctypes.CDLL(os.path.join(DLLPath, DLLName))
#     SimDeepDll.DoNMSLargeFile(pInput_diff.encode(encoding='UTF-8'),pAThinPath.encode(encoding='UTF-8'))


def extract_plot(i_path_1, i_path_2):
    # print(i_path_1)
    rootPath = os.path.abspath(os.path.join(i_path_1, '../..'))
    # print('-'*20)
    # print(rootPath)
    DLLPath = r"D:\lsy\tools\x64-20230913\x64\Release"  # 封装好的dll所在文件夹路径
    DLLName = "SimDeep.dll"

    pInput = i_path_1 # diff结果的路径
    pSS = i_path_2  # 语义分割结果（swin）路径


    # print('*' * 20)
    # print(rootPath)
    # if not os.path.exists(os.path.join(rootPath, 'result')):
    #     os.mkdir(os.path.join(rootPath, 'result'))
        # print('+'*20., 'success')

    a = pInput.split('\\')[-2]
    dir_name = a.split('.')[0]
    resultPath = os.path.join(rootPath, dir_name, 'result')
    if not os.path.exists(resultPath):
        os.mkdir(resultPath)
    pTempPath = os.path.join(resultPath, r'temp')  # 存放临时文件路径
    pAThinPath = os.path.join(resultPath, r'Thin')  # 细化之后的路径
    pAThinRes = os.path.join(resultPath, r'ThinRes')  # 细化之后结果的路径
    pRetPath = os.path.join(resultPath, r'Ret')  # 结果存放路径，该结果不带地理坐标
    pRetGEOPath = os.path.join(resultPath, r'RetGEO')  # 最终结果的存放路径

    if not os.path.exists(pTempPath):
        os.mkdir(pTempPath)
        os.mkdir(pAThinPath)
        os.mkdir(pAThinRes)
        os.mkdir(pRetPath)
        os.mkdir(pRetGEOPath)
    img_name = os.listdir(pInput)
    pInput = pInput + '\\' + img_name[0]
    pSS = pSS + '\\' + img_name[0]
    #ExtractField(DLLPath, DLLName, pInput, pSS, pTempPath, pAThinPath, pRetPath, pRetGEOPath)
    s = pInput.split('\\')[-1]
    result_name = s.split('.')[0] + '_RetGEO.shp'
    print(os.path.join(pRetGEOPath, result_name))
    # area(os.path.join(pRetGEOPath, result_name))

if __name__ == '__main__':

    # rootPath = r'E:\zyk\xj\xj_2019'
    rootPath = r"F:\zyk\2023"
    DLLPath = r"C:\Users\zyk12\Desktop\get_shp_0508\x64\Release"  # 封装好的dll所在文件夹路径
    DLLName = "SimDeep.dll"

    tif_names = [i for i in os.listdir(os.path.join(rootPath, 'diff-0806')) if i.endswith('.tif')]
    # tif_names = [i for i in os.listdir(os.path.join(rootPath, 'edter')) if i.endswith('.tif')]
    # tif_names = tif_names[6:7]
    print(tif_names)

    for name in tqdm(tif_names):
        # pInput = os.path.join(rootPath, r'diff_demo921_trans(gaijinxj_0806)' + '\\' + name)    # BDCN结果的路径
        # pSS = os.path.join(rootPath, r'swin0910' + '\\' + name)       # 语义分割结果（Deeplab V3+）路径
        # pInput_diff = os.path.join(rootPath, r'ddiff_demo921_trans(gaijinxj_0806)' + '\\' + name)    # diff结果的路径
        pInput = os.path.join(rootPath, r'diff-0806' + '\\' + name)  # BDCN结果的路径
        pSS = os.path.join(rootPath, r'swin_1019' + '\\' + name)  # 语义分割结果（Deeplab V3+）路径
        pInput_diff = os.path.join(rootPath, r'diff-0806' + '\\' + name)  # diff结果的路径

        # pInput = os.path.join(rootPath, r'edter' + '\\' + name)  # BDCN结果的路径
        # pSS = os.path.join(rootPath, r'swin' + '\\' + name)  # 语义分割结果（Deeplab V3+）路径


        if not os.path.exists(os.path.join(rootPath, 'result_1115')):
            os.mkdir(os.path.join(rootPath, 'result_1115'))

        dir_name = name.split('.')[0]
        resultPath = os.path.join(rootPath, 'result_1030' + '\\' + dir_name)
        if not os.path.exists(resultPath):
            os.mkdir(resultPath)
        pTempPath = os.path.join(resultPath, r'temp')  # 存放临时文件路径
        pAThinPath = os.path.join(resultPath, r'Thin')  # 细化之后的路径
        pAThinRes=os.path.join(resultPath, r'ThinRes')  # 细化之后的结果
        pRetPath = os.path.join(resultPath, r'Ret')      # 结果存放路径，该结果不带地理坐标
        pRetGEOPath = os.path.join(resultPath, r'RetGEO') # 最终结果的存放路径

        if not os.path.exists(pTempPath):
            os.mkdir(pTempPath)
            os.mkdir(pAThinPath)
            os.mkdir(pAThinRes)
            os.mkdir(pRetPath)
            os.mkdir(pRetGEOPath)

        pAThinPath=os.path.join(pAThinPath, name)
        ExtractField1(DLLPath, DLLName, pInput, pSS, pTempPath, pAThinPath, pRetPath, pRetGEOPath, pAThinRes)
        ExtractField2(DLLPath, DLLName, pInput, pSS, pTempPath, pAThinPath, pRetPath, pRetGEOPath, pAThinRes)

        # # ExtractField1(DLLPath, DLLName, pInput, pSS, pTempPath, pAThinPath, pRetPath, pRetGEOPath, pAThinRes)
        # ExtractField2(DLLPath, DLLName, pInput, pSS, pTempPath, pInput_diff, pRetPath, pRetGEOPath, pAThinRes)

        # thread1 = threading.Thread(target=ExtractField1,args=(DLLPath, DLLName, pInput, pSS, pTempPath, pAThinPath, pRetPath, pRetGEOPath,pAThinRes))
        # thread2 = threading.Thread(target=ExtractField2,args=(DLLPath, DLLName, pInput, pSS, pTempPath, pAThinPath, pRetPath, pRetGEOPath,pAThinRes))


        # # 启动第一个线程
        # thread1.start()
        #
        #
        # # 等待第一个线程结束
        # thread1.join()
        # thread2.start()
        # thread2.join()

        s = pInput.split('\\')[-1]
        result_name = s.split('.')[0] + '.shp'
        print(os.path.join(pRetGEOPath, result_name))
        # area(os.path.join(pRetGEOPath, result_name))
