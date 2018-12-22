from flask import Flask, render_template, request
# from flask.ext.restful import Api, Resource
import os
import sys
sys.path.append('..')
import depthmodel
import random

app = Flask(__name__)

root_path = os.getcwd()

# config
uploads_path = os.path.join(root_path, 'static/uploads/')
test_file_list = os.path.join(root_path, 'static/uploads/test_files_eigen.txt')
checkpoint_path = os.path.join(os.path.dirname(root_path), "models/model_city2kitti.meta")
sess, params, model = depthmodel.init(uploads_path, test_file_list, checkpoint_path)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/result", methods=['POST','GET'])
def result(image_for_query=None, file_name_pr='result'):
    data_path = uploads_path
    filenames_file = test_file_list
    output_directory = os.path.join(root_path, "static/depthmaps")
    depthmodel.test(sess, params, model, data_path, filenames_file, output_directory, file_name_pr)
    datas={
        'default_image' : image_for_query,
        'depth_map' : '/static/depthmaps/' + file_name_pr + '.png',
        'depth_map_pp' : '/static/depthmaps/' + file_name_pr + '.png'
    }
    return render_template("result.html", datas=datas)

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        file = request.files['input-1']
        if file and allowed_file(file.filename):
            filename = file.filename
            print("Get an Image file: ", filename)
            file_name_pr = str(random.randint(1, 100))
            filename = file_name_pr + '.jpg'
            image_for_query = os.path.join('/static/uploads', filename)

            default_image = image_for_query
            f = open(test_file_list, "w+")
            f.write(str(filename) + "\n")
            f.close()
        
            file.save(os.path.join(uploads_path, filename))

        return result(image_for_query, file_name_pr)
        
    return render_template('fileupload.html')

if __name__ == '__main__':
    app.run(host="", debug=False, threaded=True, port=6011)