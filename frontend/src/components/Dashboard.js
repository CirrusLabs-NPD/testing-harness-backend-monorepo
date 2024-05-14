import React, { useState } from "react";
import Container from "react-bootstrap/Container";
import Form from "react-bootstrap/Form";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Card from "react-bootstrap/Card";
import Button from "react-bootstrap/Button";
import axios from "axios";

function Dashboard() {
  const [selectedModels, setSelectedModels] = useState([]);
  const [selectedDatasets, setSelectedDatasets] = useState([]);
  const [customDataset, setCustomDataset] = useState("");
  const [datasets, setDatasets] = useState(["Standard", "Medical"]);

  const [bleuUrl, setBleuUrl] = useState("");
  const [terUrl, setTerUrl] = useState("");
  const [meteorUrl, setMeteorUrl] = useState("");
  const [accuracyUrl, setAccuracyUrl] = useState("");

  const handleCheckboxChangeModels = (event) => {
    const { value, checked } = event.target;
    if (checked) {
      setSelectedModels((prevSelectedModels) => [...prevSelectedModels, value]);
    } else {
      setSelectedModels((prevSelectedModels) =>
        prevSelectedModels.filter((model) => model !== value)
      );
    }
  };

  const handleCheckboxChangeDatasets = (event) => {
    const { value, checked } = event.target;
    if (checked) {
      setSelectedDatasets((prevSelectedDatasets) => [
        ...prevSelectedDatasets,
        value,
      ]);
    } else {
      setSelectedDatasets((prevSelectedDatasets) =>
        prevSelectedDatasets.filter((dataset) => dataset !== value)
      );
    }
  };
  const handleInputChange = (e) => {
    setCustomDataset(e.target.value);
  };

  const handleAddCustomDataset = () => {
    if (customDataset) {
      setDatasets([...datasets, customDataset]);
      setCustomDataset("");
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      setBleuUrl("");
      setTerUrl("");
      setMeteorUrl("");
      setAccuracyUrl("");

      // Make API call to backend
      const response = await axios.post(
        "http://localhost:8000/core/test_dataset/",
        {
          selected_models: selectedModels,
          datasets: selectedDatasets,
        }
      );

      const { bleu_url, ter_url, accuracy_url } = response.data;

      setBleuUrl(`http://localhost:8000${bleu_url}`);
      setTerUrl(`http://localhost:8000${ter_url}`);
      // setMeteorUrl(`http://localhost:8000${meteor_url}`);
      setAccuracyUrl(`http://localhost:8000${accuracy_url}`);

      console.log("bleu_image:", bleuUrl);
      console.log("ter_image:", terUrl);
      // console.log("meteor_image:", meteorUrl);
      console.log("accuracy_image:", accuracyUrl);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <Container>
      <div className="px-4">
        <Row>
          <div className="text-center">
            <h1>Machine Translation</h1>
            <Form onSubmit={handleSubmit}>
              <Col style={{ alignItems: "center" }}>
                <h2 className="h5">Choose dataset(s):</h2>
                {datasets.map((dataset, index) => (
                  <Form.Check
                    key={index}
                    inline
                    label={dataset}
                    name="dataset"
                    type="checkbox"
                    value={dataset}
                    checked={selectedDatasets.includes(dataset)}
                    onChange={handleCheckboxChangeDatasets}
                    id={`checkbox-dataset-${index}`}
                  />
                ))}
              </Col>
              <Col style={{ alignItems: "center" }}>
                <h2 className="h5">Models to test against:</h2>
                <Form.Check
                  inline
                  label="Google T5"
                  name="models"
                  type="checkbox"
                  value="Google T5"
                  checked={selectedModels.includes("Google T5")}
                  onChange={handleCheckboxChangeModels}
                  id="checkbox-model-1"
                />
                <Form.Check
                  inline
                  label="Facebook NLLB"
                  name="models"
                  type="checkbox"
                  value="Facebook NLLB"
                  checked={selectedModels.includes("Facebook NLLB")}
                  onChange={handleCheckboxChangeModels}
                  id="checkbox-model-2"
                />
                <Form.Check
                  inline
                  label="Helsinki Opus"
                  name="models"
                  type="checkbox"
                  value="Helsinki Opus"
                  checked={selectedModels.includes("Helsinki Opus")}
                  onChange={handleCheckboxChangeModels}
                  id="checkbox-model-3"
                />
              </Col>
              <Col style={{ alignItems: "center" }}>
                <Button variant="primary" type="submit">
                  Test
                </Button>
              </Col>
            </Form>
          </div>
        </Row>
      </div>
      <div className="px-4 pt-4">
        <Row className="mx-auto">
          {bleuUrl && (
            <Col>
              <Card className="mx-auto" style={{ width: "18rem" }}>
                <Card.Img variant="top" src={bleuUrl} />
                <Card.Body>
                  <Card.Title>BLEU Plot</Card.Title>
                  <Card.Text>
                    BLEU scores measure the similarity of the machine-translated
                    text to a set of high quality reference translations.
                  </Card.Text>
                </Card.Body>
              </Card>
            </Col>
          )}
          {terUrl && (
            <Col>
              <Card className="mx-auto" style={{ width: "18rem" }}>
                <Card.Img variant="top" src={terUrl} />
                <Card.Body>
                  <Card.Title>TER Plot</Card.Title>
                  <Card.Text>
                    TER scores measure the number of edit operations needed to
                    transform the machine-translated output into a human
                    translated reference
                  </Card.Text>
                </Card.Body>
              </Card>
            </Col>
          )}
          {meteorUrl && (
            <Col>
              <Card className="mx-auto" style={{ width: "18rem" }}>
                <Card.Img variant="top" src={meteorUrl} />
                <Card.Body>
                  <Card.Title>METEOR Plot</Card.Title>
                  <Card.Text>
                    METEOR scores measure the quality of generated text based on
                    the alignment between the generated text and the reference
                    text.
                  </Card.Text>
                </Card.Body>
              </Card>
            </Col>
          )}
          {accuracyUrl && (
            <Col>
              <Card className="mx-auto" style={{ width: "18rem" }}>
                <Card.Img variant="top" src={accuracyUrl} />
                <Card.Body>
                  <Card.Title>ACCURACY Plot</Card.Title>
                  <Card.Text>
                    Accuracy is the proportion of correct predictions among the
                    total number of cases processed. It can be computed with:
                    Accuracy = (TP + TN) / (TP + TN + FP + FN)
                  </Card.Text>
                </Card.Body>
              </Card>
            </Col>
          )}
        </Row>
      </div>
    </Container>
  );
}

export default Dashboard;
