from flask import Blueprint, request,jsonify,redirect
import bert.module.bertTyai as bertTyaiModel
bertCtrl = Blueprint('bert-tyai',__name__)

@bertCtrl.route('', methods=['GET','POST'])
def getResult():
  if request.method=='GET':
    return jsonify({'result':'請使用POST形式傳入文字。'})
  elif request.method=="POST":
    json = request.get_json(True)
    text = json['text']
    result = bertTyaiModel.getResult(text[:2],text[3:])
    return jsonify({'result':result})
