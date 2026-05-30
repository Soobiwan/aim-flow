from PIL import Image

from aim_flow.visualize import make_comparison_gallery


def test_make_comparison_gallery_builds_labeled_rows(tmp_path):
    paths = []
    for index in range(4):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (8, 6), (index * 30, 0, 0)).save(path)
        paths.append(path)

    output = make_comparison_gallery(
        image_rows=[paths[:2], paths[2:]],
        column_labels=["base", "method"],
        row_labels=["first", "second"],
        output_path=tmp_path / "gallery.png",
    )

    assert output.exists()
    assert Image.open(output).size == (266, 70)
