import config
from flask import Flask,request,render_template
from flask_cors import CORS
from flask_flatpages import FlatPages
from bert.controller.bertTyai import bertCtrl


app=Flask(__name__)
CORS(app)
app.config.from_object(config) # 由config.py管理環境變數
app.register_blueprint(bertCtrl, url_prefix='/bert-tyai')
pages = FlatPages(app)
