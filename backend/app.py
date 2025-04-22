from datetime import datetime
import pytz
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, request, jsonify, session, current_app, url_for, send_from_directory
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.xception import preprocess_input
import numpy as np
from keras.src.layers import Dense, GlobalAveragePooling2D, Reshape, GlobalMaxPooling2D, Lambda, Concatenate, Conv2D, \
    Add, Activation, Multiply
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
from models import db, User, Image
from config import Config
from keras import backend as K

# 创建Flask应用实例，并启用CORS，这样前端应用即使部署在不同的域也可以与之通信。
app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config.from_object(Config)
app.config['SECRET_KEY'] = 'hello world'
db.init_app(app)


def custom_mean(x):
    return K.mean(x, axis=3, keepdims=True)


def custom_max(x):
    return K.max(x, axis=3, keepdims=True)

# 加载模型
# 确保在加载模型时，将这些自定义函数传递给custom_objects参数
model_path = ('C:/Users/cissy/Desktop/models/my_model_further_epoch_04_val_accuracy_0.90.h5')
model = load_model(model_path, custom_objects={'K': K, 'custom_mean': custom_mean, 'custom_max': custom_max})


# @app.before_first_request
# def create_tables():
#     db.create_all()
# 自动创建数据库

def get_beijing_time():
    utc_now = datetime.utcnow()  # 获取当前的UTC时间
    utc_now = utc_now.replace(tzinfo=pytz.utc)  # 设置UTC时间的时区
    beijing_tz = pytz.timezone('Asia/Shanghai')  # 定义北京时区
    beijing_now = utc_now.astimezone(beijing_tz)  # 转换到北京时区
    return beijing_now


@app.after_request
def cookies_secure(response):
    session_cookie = session.get('session_cookie_name')
    if session_cookie:
        response.set_cookie('session_cookie_name', session_cookie, secure=True, samesite='None')
    return response


def check_fake(image_path):
    img = image.load_img(image_path, target_size=(299, 299))  # 调整图像大小以匹配模型输入
    img_array = image.img_to_array(img)  # 将PIL图像转换为Numpy数组
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)  # 增加一个维度以创建一个形状为(1, 299, 299, 3)的数组，代表批量大小为1的图像批次
    img_preprocessed = preprocess_input(img_array_expanded_dims)  # 使用与训练时相同的预处理，确保预测时输入数据的格式和分布与训练时一致

    prediction = model.predict(img_preprocessed)

    is_fake = not (prediction > 0.5)
    return {"isFake": is_fake}


@app.route('/api/register', methods=['POST'])
def register():
    # 获取JSON请求数据
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    # 检查用户名是否已存在
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already exists'}), 400

    # 创建新用户实例
    new_user = User(username=username)
    new_user.set_password(password)

    # 添加到数据库会话并提交
    db.session.add(new_user)
    db.session.commit()

    # 保存用户ID到session中
    session['user_id'] = new_user.id
    print("Session user_id:", session.get('user_id'))

    return jsonify({'success': 'User registered'})


@app.route('/api/login', methods=['POST'])
def login():
    # 获取JSON请求数据
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()
    if user is None or not user.check_password(password):
        return jsonify({'error': 'Invalid username or password'}), 401

    session['user_id'] = user.id
    print("Session user_id:", session.get('user_id'))
    return jsonify({'success': 'Logged in successfully'})


@app.route('/api/logout', methods=['POST'])
def logout():
    # 清除会话中的用户ID
    session.pop('user_id', None)  # 使用 pop 方法移除 user_id，并防止如果不存在时报错

    return jsonify({'success': 'Logged out successfully'})


@app.route('/api/user/update', methods=['PUT'])
def update_user():
    # print("Received update request")
    # print("Session user_id:", session.get('user_id'))
    data = request.get_json()
    user_id = session.get('user_id')

    if not user_id:
        return jsonify({'error': 'User not authenticated'}), 401

    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    username = data.get('username')
    password = data.get('password')

    if username and User.query.filter(User.username == username, User.id != user_id).first():
        return jsonify({'error': 'Username already exists'}), 400
    if username:
        user.username = username

    if password:
        user.set_password(password)

    db.session.commit()

    return jsonify({'success': 'User updated successfully'})


@app.route('/api/history', methods=['GET'])
def view_history():
    # 验证用户是否已登录
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Authentication required'}), 401

    # 查询数据库，获取当前用户的所有检测记录
    user_images = Image.query.filter_by(user_id=user_id).all()

    # 将查询结果转换成JSON格式
    history_records = []
    for img in user_images:
        record = {
            'ID': img.id,
            'filename': img.filename,
            'url': img.url,
            'is_fake': img.is_fake,
            'uploaded_at': img.uploaded_at.strftime('%Y-%m-%d %H:%M:%S'),  # 格式化日期时间
            'userID': img.user_id
        }
        history_records.append(record)

    # 返回用户的历史检测记录
    return jsonify(history_records)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    # 验证用户是否已登录
    user_id = session.get('user_id')
    # if not user_id:
    #     return jsonify({'error': 'Authentication required'}), 401

    # 检查是否有文件在请求中
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # 获取北京时间
        now = get_beijing_time()
        # 使用当前时间戳创建唯一文件名
        timestamp = now.strftime('%Y%m%d%H%M%S')
        # 使用secure_filename清理文件名并添加时间戳
        filename = secure_filename(f"{timestamp}_{file.filename}")
        # 确保UPLOAD_FOLDER目录存在
        upload_folder = current_app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)

        result = check_fake(filepath)


        # 创建图片的访问URL
        image_url = url_for('uploaded_file', filename=filename, _external=True)

        # 将检测结果保存到数据库
        # new_image = Image(filename=filename, is_fake=result['isFake'], uploaded_at=datetime.utcnow(), user_id=user_id,
        #                   url=image_url)
        # db.session.add(new_image)
        # db.session.commit()
        if user_id:
            new_image = Image(filename=filename, is_fake=result['isFake'], uploaded_at=now,
                              user_id=user_id, url=image_url)
            db.session.add(new_image)
            db.session.commit()

        # return jsonify(result)
        return jsonify({
            'isFake': result['isFake'],
            'image_url': image_url
        })


if __name__ == '__main__':
    app.run(debug=True)
