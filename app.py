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
beta = [0, 0]
app = Flask(__name__)


@app.route('/')
def home():
    html = "<h3>Linear Regression Model Home</h3>"
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
