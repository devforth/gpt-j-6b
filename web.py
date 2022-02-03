from pandas import datetime
import cherrypy
import datetime
print('Before model import')
before = datetime.datetime.now()
import model

class MyWebService(object):

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def process(self):
        data = cherrypy.request.json
        output = {"output": model.eval(data)}
        return output

print('Inititalizing, model imported within', datetime.datetime.now() - before)

if __name__ == '__main__':
    config = {'server.socket_host': '0.0.0.0'}
    print('Binding to 0.0.0.0')
    cherrypy.config.update(config)
    cherrypy.quickstart(MyWebService())
