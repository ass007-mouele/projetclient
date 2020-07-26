from flask_script import Manager , Server
from flask_migrate import Migrate, MigrateCommand
from app import app, db


migrate = Migrate(app, db)
manager = Manager(app)

manager.add_command('db', MigrateCommand)
manager.run()

#if __name__ == '__main__':
    #manager.run()

#manager.add_command('runserver', Server(host='localhost', port=8080, debug=True))
