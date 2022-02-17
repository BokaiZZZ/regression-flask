from flask import Flask, request, jsonify, render_template, Response
from model import regression
from regression import RegressionAlgorithm
from ridge import Ridge
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from generate_data import generate_data
from lasso import Lasso
import numpy as np
import io


cv = 5
model = Lasso()
model_dict = {"Linear": RegressionAlgorithm(),
              "Lasso": Lasso(),
              "Ridge": Ridge()}
app = Flask(__name__)


@app.route('/')
def home():
    html = "<h3>Welcome to Linear Regression Model Home</h3>"
    return html.format(format)


@app.route('/show', methods=['POST', 'GET'])
def show():
    if request.method == 'POST':
        reg_type = request.form.get("type")
        global cv
        global model
        cv = request.form.get("K")
        model = model_dict.get(reg_type)
        return render_template('regression_plot.html')
    return render_template('regression_page.html')


@app.route('/plot.png', methods=['GET'])
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    X, y = generate_data()
    c = np.ones(X.shape[1])
    rel = regression(model, X, y, method="Robust",
                 c=c, K=int(cv), criterion="MSE")
    y_pred = rel.predict(X)
    axis.scatter(X, y, color='black')
    axis.plot(X, y_pred, color='blue')
    axis.legend(("Fitted Line", "Points"))
    return fig


@app.route('/build_model', methods=['POST'])
def build_model():
    """Performs an sklearn prediction
    input looks like:
    {'model': 'Lasso',
     'X': [[0.5574711078885419],
      [0.5517481173235443],
      [0.14345970630378713],
      [1.8961842529495423],
      [0.046132050637789836],
      [1.027253794196982],
      [-0.024801038825939852],
      [-0.598052738423797],
      [-1.0702306427835442],
      [-1.898078499394308],
      [-0.9019300246798666],
      [-0.1155756107669977],
      [-0.469828721963862],
      [-0.4674348447745329],
      [-0.5845275300993473],
      [-0.35831795707600933],
      [-0.9768328174988236],
      [1.124320647217659],
      [0.5036755173769516],
      [-0.010711840064757202]],
     'y': [5.095977952500841,
      8.616147858452027,
      5.536614442248911,
      11.239200780448671,
      4.440353419890255,
      9.180826579444602,
      5.152691641838957,
      0.9401332834519618,
      0.05306440207573759,
      -3.1795710555150554,
      -0.746149626022764,
      3.03729090313729,
      1.815419679151728,
      1.9956518972995578,
      -0.3266346935578839,
      3.271430383690231,
      1.7833785150055537,
      7.590932471756622,
      7.242838352592065,
      3.4104872778257094]}
    Note that it is also possible to have M, K, method, c, criterion etc.
    More details can be viewed in description of regression() in model.py 
    result looks like:
    {'coefficient': [[4.032748962682868], [4.147988972861122]]}
    """
    content = request.get_json()
    model_type = model_dict.get(content["model"])
    X = np.array(content["X"])
    y = np.array(content["y"])
    del content["model"]
    del content["X"]
    del content["y"]
    rel = regression(model_type, X, y, **content)
    return jsonify({"coefficient": rel.beta.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
