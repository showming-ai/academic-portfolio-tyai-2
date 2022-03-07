from bert import app
import config

@app.route('/')
def index():
  return 'server started on '+str(config.PORT)+' PORT '+str(config.ENV)
  
if __name__ == '__main__':
  print(app.url_map)
  app.run(host = config.HOST,port=config.PORT,debug=True)
