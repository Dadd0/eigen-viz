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
    return Matrix, alt, mo, np, pd, plt


@app.cell
def _(Matrix, mo):
    A = mo.ui.anywidget(Matrix(cols=2, rows=2, matrix=[[1,0], [0,1]]))
    return (A,)


@app.cell
def _(A, np):
    eigenvalues, eigenvectors = np.linalg.eig(np.array(A.value["matrix"]))
    eigenvalues = eigenvalues.round(2)
    return eigenvalues, eigenvectors


@app.cell
def _(mo):
    mo.md(
        r"""
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
                """$A_{2,2} =$"""),
            A
        ], 
        justify="center", align="center"
    )
    return


@app.cell
def _(eigenvalues, mo):
    mo.center(mo.md(
                    rf"""
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
    np,
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
    arrow_layer = alt.Chart(df_arrows).mark_rule(strokeWidth=4).encode(
        x='x:Q',
        y='y:Q',
        x2='x2:Q',
        y2='y2:Q',
        color=alt.Color('color:N', scale=None),
    )

    df_arrows["angle"] = np.degrees(np.arctan2(df_arrows["y2"], df_arrows["x2"]))
    arrow_heads = alt.Chart(df_arrows).mark_point(
        shape="triangle", filled=True, fillOpacity=1, size=100
    ).encode(
        x='x2:Q',
        y='y2:Q',
        angle='angle:Q',
        color=alt.Color('color:N', scale=None)
    )

    label_layer = alt.Chart(df_arrows).mark_text(dx=5, dy=-5, fontSize=14, fontWeight="bold").encode(
        x='x2:Q',
        y='y2:Q',
        text='label:N',
        color=alt.Color('color:N', scale=None)
    )


    chart_2 = (points_tfmd + x_axis + y_axis + arrow_layer + arrow_heads + label_layer).properties(
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    <hr>
    ## Visualization in $\mathbb{R^3}$
    """
    )
    return


@app.cell
def _(Matrix, mo):
    A_3d = mo.ui.anywidget(Matrix(cols=3, rows=3, matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    return (A_3d,)


@app.cell
def _(A_3d, mo):
    mo.hstack(
        [
            mo.md(
                """$A_{3,3} =$"""),
            A_3d
        ], 
        justify="center", align="center"
    )
    return


@app.cell
def _(eigenvalues_3d, mo):
    mo.center(mo.md(
                    rf"""
                    **Eigenvalues**:
                        [$\lambda_1={eigenvalues_3d[0]}$
                        $\lambda_2={eigenvalues_3d[1]}$,
                        $\lambda_3={eigenvalues_3d[2]}$]
                        """
    ))
    return


@app.cell(hide_code=True)
def _(np, plt):
    x = np.linspace(-12, 12, 13)
    y = np.linspace(-12, 12, 13)
    z = np.linspace(-12, 12, 13)
    X, Y, Z = np.meshgrid(x, y, z)
    original_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])

    fig_original = plt.figure(figsize=(8, 6))
    ax_original = fig_original.add_subplot(111, projection='3d')
    ax_original.scatter(original_points[0, :], original_points[1, :], original_points[2, :], c="#3B0270", alpha=0.6)
    ax_original.set_xlabel('X')
    ax_original.set_ylabel('Y')
    ax_original.set_zlabel('Z')
    ax_original.set_title("Original Datapoints", fontsize=9, fontweight="bold");
    return fig_original, original_points


@app.cell
def _(np, original_points, plt):
    def plot_transformed_points_and_eigenvectors(matrix_widget):
            matrix = np.array(matrix_widget.value["matrix"])

            transformed_points = original_points.T @ matrix
            eigenvalues, eigenvectors = np.linalg.eig(matrix)

            fig_transformed = plt.figure(figsize=(8, 6))
            ax_transformed = fig_transformed.add_subplot(111, projection='3d')
            ax_transformed.scatter(transformed_points[:, 0], 
                                  transformed_points[:, 1], 
                                  transformed_points[:, 2], 
                                  c="#FEB21A", 
                                  alpha=0.3, 
                                  s=10,        
                                  depthshade=True)  

            colors = ['#ED3F27', '#134686', '#28A745']
            for i in range(3):
                vec = eigenvalues[i] * eigenvectors[:, i] * 10  


                ax_transformed.quiver(0, 0, 0, vec[0], vec[1], vec[2], 
                                     color=colors[i], 
                                     linewidth=2,  
                                     arrow_length_ratio=0.15, 
                                     alpha=1.0,
                                     zorder=1000 + i)  

                ax_transformed.plot([0, vec[0]], [0, vec[1]], [0, vec[2]], 
                                   color=colors[i], 
                                   linewidth=2, 
                                   alpha=0.3,
                                   zorder=999 + i)

                ax_transformed.text(vec[0], vec[1], vec[2], f"λ{i+1}", 
                                   color=colors[i], 
                                   fontsize=10, 
                                   fontweight='bold',
                                   zorder=2000 + i)

            ax_transformed.scatter([0], [0], [0], 
                                  color='black', 
                                  s=100, 
                                  marker='o',
                                  edgecolors='white',
                                  linewidths=2,
                                  zorder=1100)

            ax_transformed.view_init(elev=20, azim=45)
            ax_transformed.grid(True, alpha=0.3)

            ax_transformed.set_title("Transformed Data", fontsize=9, fontweight="bold")
            ax_transformed.set_xlabel('X')
            ax_transformed.set_ylabel('Y')
            ax_transformed.set_zlabel('Z')

            return fig_transformed, np.round(eigenvalues, 2)
    return (plot_transformed_points_and_eigenvectors,)


@app.cell
def _(A_3d, plot_transformed_points_and_eigenvectors):
    fig_transformed, eigenvalues_3d = plot_transformed_points_and_eigenvectors(A_3d)
    return eigenvalues_3d, fig_transformed


@app.cell
def _(fig_original, fig_transformed, mo):
    mo.hstack(
        [
            fig_original,
            fig_transformed
        ], justify="center"
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
