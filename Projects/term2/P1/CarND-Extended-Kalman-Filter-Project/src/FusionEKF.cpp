#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  /**
    TODO:
      * Finish initializing the FusionEKF.
      * Set the process and measurement noises
   */
  H_laser_ << 1, 0, 0, 0,
            0, 1, 0, 0;

  noise_ax = 9;
  noise_ay = 9;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
    VectorXd x_init(4);
    x_init << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      format: void polar_to_cartesian(const MeasurementPackage &measurement_pack, double &px, double &py)
      */
      float px, py;
      float rho, theta, rho_dot;
      rho = measurement_pack.raw_measurements_[0];
      theta = measurement_pack.raw_measurements_[1];
      std::cout << "rho, theta: " << rho << ", " << theta << std::endl;

      px = rho * cos(theta);
      py = rho * sin(theta);
      x_init << px, py, 0.0, 0.0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      x_init << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    // initialize Kalman Filter
    MatrixXd P_init(3, 3);
    MatrixXd F_init(3, 3);
    P_init << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;
    F_init << 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;

    ekf_.Init(x_init, P_init, F_init);

    // save timestamp
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;

    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
   // get time diff
   float elapsed = (measurement_pack.timestamp_ - previous_timestamp_) / 1.0e6;
   std::cout << "elapsed: " << elapsed << std::endl;
   // update F
   ekf_.F_(0, 2) = elapsed;
   ekf_.F_(1, 3) = elapsed;

   // update Q
   float e2 = std::pow(elapsed, 2);
   float e3 = std::pow(elapsed, 3);
   float e4 = std::pow(elapsed, 4);
   ekf_.Q_ << noise_ax * e4 / 4.0f, 0, noise_ax * e3 / 2.0f, 0,
            0, noise_ay * e4 / 4.0f, 0, noise_ay * e3 / 2.0f,
            noise_ax * e3 / 2.0f, 0, noise_ax * e2, 0,
            0, noise_ay * e3 / 2.0, 0, noise_ay * e2;


  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    // calculate Hj_ from H_radar
    Hj_ = tools.CalculateJacobian(ekf_.x_);
    // update H_ and R_
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    // update H_ and R_
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
