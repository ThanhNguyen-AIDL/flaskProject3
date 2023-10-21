import os
import time
# import cv2
import numpy
import secrets
from PIL import Image
from flask import render_template, url_for, flash, redirect, request, send_from_directory, send_file
from flaskblog import app, db, bcrypt
from flaskblog.forms import RegistrationForm, LoginForm, UpdateAccountForm
from flaskblog.models import User, Post
from flask_login import login_user, current_user, logout_user, login_required
from flaskblog.service import generateImageBasedOnText
from werkzeug.utils import secure_filename

# from flaskblog.service.OCR.main import main as read_id_card

app.config["image_account"] = "static\profile_pics"
app.config["image_upload_original"] = "static\database\id_card\image\image_original"
app.config["image_upload_processed"] = "static/database/id_card/image/image_processed/"
app.config["size_image_account"] = (125, 125)
app.config["size_image_id-card"] = (500, 300)
posts = [
    {
        'author': 'thanh nguyen ',
        'title': 'Blog Post 1',
        'content': 'First post content',
        'date_posted': 'April 20, 2018'
    },
    {
        'author': 'thanh nguyen2',
        'title': 'Blog Post 2',
        'content': 'Second post content',
        'date_posted': 'April 21, 2018'
    }
]


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', posts=posts)


@app.route("/about")
def about():
    return render_template('about.html', title='About')


@app.route("/Sign_Up", methods=['GET', 'POST'])
def Sign_Up():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('your account {} has been created!'.format(form.username.data), 'success')
        return redirect(url_for('Sign_In'))
    return render_template('sign_up.html', title='Sign_Up', form=form)


@app.route("/Sign_In", methods=['GET', 'POST'])
def Sign_In():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check email and password ', 'danger')
    return render_template('sign_in.html', title='Sign_In', form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))


def save_picture(form_picture, path_image, output_size):
    random_hex = secrets.token_hex(8)
    f_name, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, path_image, picture_fn)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)
    return picture_fn


@app.route("/account", methods=['GET', 'POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data, app.config["image_account"],
                                        app.config["size_image_account"])
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('You account has been updated!', 'success')
        return redirect(url_for('account'))
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('Account.html', title='Account', image_file=image_file, form=form)


@app.route("/Id_card", methods=['GET', 'POST'])
def Id_card():
    result = []
    if request.method == "POST":
        if request.files:
            image_file = request.files["image"]
            if len(request.form) == 0:
                flash('vui lòng chọn loại thẻ cần trích xuất', 'danger')
                return redirect(url_for("Id_card"))
            else:
                req = request.form["id-card"]
            if image_file.filename == '':
                flash('vui lòng chọn tệp cần tải lên', 'danger')
                return redirect(url_for("Id_card"))
            # get result of id-card
            if req == 'cancuoc':
                card_type = '3'
            elif req == 'cmnd-new':
                card_type = '2'
            else:
                card_type = '1'
            # img = cv2.imdecode(numpy.fromstring(image_file.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
            # result, image = read_id_card(img, card_type)
            image_original = save_picture(image_file, app.config["image_upload_original"],
                                          app.config["size_image_id-card"])
            # cv2.imwrite('flaskblog/static/database/id_card/image/image_processed/' + image_original, image)
            file_image = url_for('static', filename='database/id_card/image/image_processed/' + image_original)
            return render_template('Id_card.html', title='Id_card', filename=file_image, result=result,
                                   card_type=card_type)
    return render_template('Id_card.html', title='Id_card', result=result)


@app.route("/image_generate", methods=['GET', 'POST'])
def image_generate():
    result = ''
    if request.method == "POST":
        prompt = "So what are you waiting for? Call ABC Company today at 1-234-567-890 and schedule a time to meet our adorable dogs!"
        if request.form.get('prompt') is not None:
            prompt = request.form.get('prompt')

        url, headers, payload = generateImageBasedOnText.initial_configs()
        response = generateImageBasedOnText.text_to_image(url, headers, payload, prompt)
        # save images that generated
        images = generateImageBasedOnText.save_results(response)
        return render_template('images.html', title='image_generate', images=images)
    return render_template('images.html', title='image_generate', images=None)


@app.route('/serve_image/<image_id>')
def serve_image(image_id):
    print("sdlkfjsd")
    return send_file(image_id, mimetype='image/png')
