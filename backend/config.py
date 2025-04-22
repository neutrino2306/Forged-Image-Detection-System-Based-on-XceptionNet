class Config:
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:521592@localhost/forgedimagesdetection'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = 'your_secret_key'
    UPLOAD_FOLDER = 'uploads'