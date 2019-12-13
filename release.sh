#coding:utf-8
###################################################
# File Name: release.sh
# Author: Meng Zhao
# mail: @
# Created Time: 2019年12月13日 星期五 14时26分04秒
#=============================================================
mkdir dist
mkdir dist/data
cp setting.py dist/

cp -r data/stopword_data dist/data

cp -r output dist/

cp -r web dist/


cp -r preprocess dist/

cp -r common dist/

