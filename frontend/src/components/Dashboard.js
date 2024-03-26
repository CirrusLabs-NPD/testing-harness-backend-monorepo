import React from "react";
import Container from "react-bootstrap/Container";
import Form from "react-bootstrap/Form";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Button from "react-bootstrap/Button";

function Dashboard() {
  return (
    <Container>
      <div className="px-4">
        <Row>
          <Col style={{ alignItems: "center" }}>
            <h2 className="h5">Choose a dataset:</h2>
            <Form>
              <Form.Check
                inline
                label="Standard"
                name="inlineRadioOptions"
                type="radio"
                id="radio-model-1"
              />
              <Form.Check
                inline
                label="Medical"
                name="inlineRadioOptions"
                type="radio"
                id="radio-model-2"
              />
            </Form>
          </Col>
          <Col style={{ alignItems: "center" }}>
            <h2 className="h5">Models to test against:</h2>
            <Form>
              <Form.Check
                inline
                label="Google T5"
                name="inlineRadioOptions"
                type="checkbox"
                id="checkbox-model-1"
              />
              <Form.Check
                inline
                label="Facebook NLLB"
                name="inlineRadioOptions"
                type="checkbox"
                id="checkbox-model-2"
              />
              <Form.Check
                inline
                label="Helsinki Opus"
                name="inlineRadioOptions"
                type="checkbox"
                id="checkbox-model-3"
              />
            </Form>
          </Col>
          <Col style={{ alignItems: "center" }}>
            <Button variant="primary" type="submit">
              Test
            </Button>
          </Col>
        </Row>
      </div>
    </Container>
  );
}

export default Dashboard;
