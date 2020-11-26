__author__ = 'Anwen Hu'

import os, gzip, cStringIO
import xml.etree.cElementTree as ET
import json
from copy import deepcopy

import sys

reload(sys)
sys.setdefaultencoding('utf8')


def check_caption(caption):
    # TODO: \u, \uxxxx
    # u'Pope Francis\u2019s charismatic New Year\u2019s sermon demonstrated how he has been able to capture hearts worldwide\n'
    if '\n' in caption:
        print('###\\n in caption###')
        print 'raw: ', caption
        # caption = raw_caption.strip().split('\n')[0].strip()
        caption = caption.replace('\n', '')
        print 'processed: ', caption
    if '<i>' in caption or '</i>' in caption:
        print('###<i> in caption###')
        print 'raw: ', caption
        caption = caption.replace('<i>', '').replace('</i>', '')
        print 'processed: ', caption
    if 'Photograph:' in caption:
        print('###Photograph: in caption###')
        print 'raw: ', caption
        caption = caption.strip().split('Photograph: ')[0].strip()
        print 'processed: ', caption
    if '<p>' in caption or '</p>' in caption:
        print('###<p> in caption###')
        print 'raw: ', caption
        caption = caption.replace('<p>', '').replace('</p>', '')
        print 'processed: ', caption
    return caption


def read_xml(path):
    """
    read each xml in BreakingNews(each xml contains multiple articles)
    :param path:
    :param cap_writer:
    :param img_writer:
    :param article_writer:
    :param cap_id:
    :param img_id:
    :return:
    """
    items = []
    item = {}
    article = {}
    xm = gzip.open(path, 'rb')
    # writer = open('test_caption.txt', 'w')
    # img_num = 0
    # captions = []
    print("read file %s" % (os.path.abspath(path)))
    for event, elem in ET.iterparse(xm, events=('start', 'end')):
        if event == 'start':
            if elem.tag == 'article':
                item['article_id'] = elem.attrib['id']
                # print('article id', elem.attrib['id'])
                item['captions'] = []
                item['images'] = []
                item['cap_ids'] = []
                item['img_ids'] = []
                article['id'] = elem.attrib['id']
                article['sentences'] = []
            if elem.tag == 'img':
                img_attribs = elem.attrib
                if 'caption' in img_attribs:
                    # TODO: ADD CHECK FOR CAPTION 2019/05/27
                    caption = img_attribs['caption']

                    if len(caption) > 0:
                        print 'red caption: ', type(caption), caption, img_attribs['file']
                        caption = check_caption(caption)
                        item['captions'].append(caption)
                        """if isinstance(caption, str):
                            caption_unicode = caption.decode('UTF-8')
                            print 'caption_unicode:', type(caption_unicode), repr(caption_unicode)
                        elif isinstance(caption, unicode):
                            caption_unicode = caption.encode('UTF-8').decode('UTF-8')
                            print 'caption_unicode:', type(caption_unicode), repr(caption_unicode)"""
                        """if isinstance(caption, unicode):
                            print(path)
                        if '\u' in caption:
                            print(path)
                            exit()"""
                    else:
                        print('null caption, img_file', img_attribs['file'])
                    # print('caption', img_attribs['caption'])
                    # print('img path', img_attribs['file'])
        elif event == 'end' and elem.tag == 'p':
            article['sentences'].append(elem.text)
            # print('sentence', elem.text)
        elif event == 'end' and elem.tag == 'article':
            # print(item['captions'])
            items.append(item)
            item = deepcopy(item)  # avoid rewriting previous items in list
            # captions.append(item['captions'])
            # img_num += len(item['images'])
    xm.close()
    # print(len(items))
    # print(img_num)
    return items


def read_BreakingNews(root_dir):
    xml_pathes = []
    month_list = os.listdir(root_dir)
    for m in range(len(month_list)):
        day_list = os.listdir(root_dir + '/' + month_list[m])
        for d in range(len(day_list)):
            xml_files = os.listdir(root_dir + '/' + month_list[m] + '/' + day_list[d])
            assert len(xml_files) == 1
            xml_path = '/' + month_list[m] + '/' + day_list[d] + '/' + xml_files[0]
            xml_pathes.append(xml_path)
    return xml_pathes


def generate_breakingnews_json(root_dir):
    xml_pathes = read_BreakingNews(root_dir)
    for xml_path in xml_pathes:
        items = read_xml(root_dir + xml_path)


if __name__ == '__main__':
    read_xml(path = "E:/files_anwenhu/NER_Experiment/MM_caption/data/BreakingNews/BreakingNewsDataset_v1.2_ArticlesStripped/data/12/10/articles_stripped.xml.gz")
    """root_dir = 'E:/files_anwenhu/NER_Experiment/MM_caption/data/BreakingNews/BreakingNewsDataset_v1.2_ArticlesStripped/data'
    generate_breakingnews_json(root_dir)"""
