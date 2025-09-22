import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium", app_title="Eigen-Viz")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import altair as alt
    import pandas as pd
    from wigglystuff import Matrix
    return Matrix, alt, mo, np, pd


@app.cell
def _(Matrix, mo):
    A = mo.ui.anywidget(Matrix(cols=2, rows=2))
    return (A,)


@app.cell
def _(A, np):
    eigenvalues, eigenvectors = np.linalg.eig(np.array(A.value["matrix"]))
    return eigenvalues, eigenvectors


@app.cell
def _(mo):
    mo.md(
        """
    # Eigenvalue and Eigenvectors Visualizer
    <hr>
    <br>

    ## Visualization in $\mathbb{R^2}$
    """
    )
    return


@app.cell
def _(A, mo):
    mo.hstack(
        [
            mo.md(
                """$A =$"""),
            A
        ], 
        justify="center", align="center"
    )
    return


@app.cell
def _(eigenvalues, mo):
    mo.center(mo.md(
                    f"""
                    **Eigenvalues**:
                        [$\lambda_1={eigenvalues[0]}$
                        $\lambda_2={eigenvalues[1]}$]
                        """
    ))
    return


@app.cell
def _(np, pd):
    x_vals = np.linspace(-10, 10, 25)
    y_vals = np.linspace(-10, 10, 25)
    grid_points = np.array([[x, y] for x in x_vals for y in y_vals])
    df_original = pd.DataFrame(grid_points, columns=['x', 'y'])
    return df_original, grid_points


@app.cell
def _(alt, df_original, pd):
    OFFSET = 0.30 # percentage offset for visualization purposes
    d_low = (df_original.min() + (df_original.min() * OFFSET)).min()
    d_high = (df_original.max() + (df_original.max() * OFFSET)).max()

    points = (
        alt.Chart(df_original)
        .mark_point(filled=True, size=30)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[d_low, d_high])),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[d_low, d_high])),
            color=alt.value("#3B0270")
        )
    )
    x_axis = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='black').encode(y='y')
    y_axis = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(color='black').encode(x='x')
    chart = (points + x_axis + y_axis).properties(
        width=300,
        height=300,
        title="Original Datapoints"
    )

    return OFFSET, chart, x_axis, y_axis


@app.cell
def _(A, grid_points, np, pd):
    transformed_points = grid_points @ np.array(A.matrix)
    df_transformed = pd.DataFrame(transformed_points, columns=['x', 'y'])
    return (df_transformed,)


@app.cell
def _(
    OFFSET,
    alt,
    chart,
    df_transformed,
    eigenvalues,
    eigenvectors,
    mo,
    pd,
    x_axis,
    y_axis,
):
    d_tfmd_low = (df_transformed.min() + (df_transformed.min() * OFFSET)).min()
    d_tfmd_high = (df_transformed.max() + (df_transformed.max() * OFFSET)).max()

    points_tfmd = (
        alt.Chart(df_transformed)
        .mark_point(filled=True, size=30)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[d_tfmd_low, d_tfmd_high])),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[d_tfmd_low, d_tfmd_high])),
            color=alt.value("#FEB21A")
        )
    )

    arrows = []
    colors = ['#ED3F27', '#134686']  
    x_span = df_transformed['x'].max() - df_transformed['x'].min()
    y_span = df_transformed['y'].max() - df_transformed['y'].min()

    for i in range(2): 
        vec = eigenvalues[i] * eigenvectors[:, i] * 7
        arrows.append({
            'x': 0, 'y': 0,
            'x2': vec[0], 'y2': vec[1],
            'color': colors[i],
            'label': f"λ{i}"
        })

    df_arrows = pd.DataFrame(arrows)
    arrow_layer = alt.Chart(df_arrows).mark_rule(strokeWidth=4, strokeDash=[6, 1.5]).encode(
        x='x:Q',
        y='y:Q',
        x2='x2:Q',
        y2='y2:Q',
        color=alt.Color('color:N', scale=None),
    )

    label_layer = alt.Chart(df_arrows).mark_text(dx=5, dy=-5, fontSize=17).encode(
        x='x2:Q',
        y='y2:Q',
        text='label:N',
        color=alt.Color('color:N', scale=None)
    )


    chart_2 = (points_tfmd + x_axis + y_axis + arrow_layer + label_layer).properties(
        width=300,
        height=300,
        title="Transformed Data"
    )


    mo.hstack(
        [
            chart,
            chart_2
        ], justify="center"
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
