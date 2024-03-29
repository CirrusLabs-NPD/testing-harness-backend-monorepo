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
  const [selectedDataset, setSelectedDataset] = useState("");
  const [bleuUrl, setBleuUrl] = useState("");
  const [terUrl, setTerUrl] = useState("");
  const [meteorUrl, setMeteorUrl] = useState("");

  const handleCheckboxChange = (event) => {
    const { value, checked } = event.target;
    if (checked) {
      setSelectedModels((prevSelectedModels) => [...prevSelectedModels, value]);
    } else {
      setSelectedModels((prevSelectedModels) =>
        prevSelectedModels.filter((model) => model !== value)
      );
    }
  };

  const handleRadioChange = (event) => {
    setSelectedDataset(event.target.value);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      setBleuUrl("");
      setTerUrl("");
      setMeteorUrl("");

      // Make API call to backend
      const response = await axios.post(
        "http://localhost:8000/core/test_dataset/",
        {
          selected_models: selectedModels,
          dataset: selectedDataset,
        }
      );

      const {
        selected_models,
        selected_dataset,
        final_results,
        bleu_url,
        ter_url,
        meteor_url,
      } = response.data;

      setSelectedModels(selected_models);
      setSelectedDataset(selected_dataset);
      setBleuUrl(`http://localhost:8000${bleu_url}`);
      setTerUrl(`http://localhost:8000${ter_url}`);
      setMeteorUrl(`http://localhost:8000${meteor_url}`);

      console.log("Selected Models:", selected_models);
      console.log("Selected Dataset:", selected_dataset);
      console.log("Final Results:", final_results);
      console.log("bleu_image:", bleuUrl);
      console.log("ter_image:", terUrl);
      console.log("meteor_image:", meteorUrl);
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
                <h2 className="h5">Choose a dataset:</h2>
                <Form.Check
                  inline
                  label="Standard"
                  name="dataset"
                  type="radio"
                  value="Standard"
                  checked={selectedDataset === "Standard"}
                  onChange={handleRadioChange}
                  id="radio-model-1"
                />
                <Form.Check
                  inline
                  label="Medical"
                  name="dataset"
                  type="radio"
                  value="Medical"
                  checked={selectedDataset === "Medical"}
                  onChange={handleRadioChange}
                  id="radio-model-2"
                />
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
                  onChange={handleCheckboxChange}
                  id="checkbox-model-1"
                />
                <Form.Check
                  inline
                  label="Facebook NLLB"
                  name="models"
                  type="checkbox"
                  value="Facebook NLLB"
                  checked={selectedModels.includes("Facebook NLLB")}
                  onChange={handleCheckboxChange}
                  id="checkbox-model-2"
                />
                <Form.Check
                  inline
                  label="Helsinki Opus"
                  name="models"
                  type="checkbox"
                  value="Helsinki Opus"
                  checked={selectedModels.includes("Helsinki Opus")}
                  onChange={handleCheckboxChange}
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
        </Row>
      </div>
    </Container>
  );
}

export default Dashboard;
