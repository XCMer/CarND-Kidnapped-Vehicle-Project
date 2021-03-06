/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // Are we debugging the predict function? If yes, then noises are not added,
    // and updateWeight & resample return without doing anything. No. of particles are also
    // hardcoded to 1.
    // https://discussions.udacity.com/t/debug-suggestion-testing-prediction-step/306124
    debug_predict = false;

    // Check if already initialized
    if (initialized()) {
        return;
    }

    // Set no. of particles
    if (debug_predict) {
        // While debugging prediction, set the no. of particles to 1
        num_particles = 1;
    } else {
        num_particles = 10;
    }

    // Initialize normal distributions
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    // Create particles
    default_random_engine gen;
    for (int i = 0; i < num_particles; i++) {
        if (debug_predict) {
            // Don't add noise in debug mode
            particles.push_back(Particle {.id=i, .x=x, .y=y, .theta=theta, .weight=1});
        } else {
            particles.push_back(Particle {.id=i, .x=dist_x(gen), .y=dist_y(gen), .theta=dist_theta(gen), .weight=1});
        }
    }

    // Set weights
    double uniform_weight = 1.0 / num_particles;
    for (int i = 0; i < num_particles; i++) {
        weights.push_back(uniform_weight);
    }

    // Set initialized
    is_initialized = true;

    // Initial state is verified
//    cout<<"Initial state: "<<particles[0].x<<", "<<particles[0].y<<", "<<particles[0].theta<<endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    // Initialize normal distributions for gaussian noise. We keep normal distribution around
    // 0 and the std dev provided to us.
    default_random_engine gen;
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    // Loop through all the particles
    for (auto &particle : particles) {
        // Predict next state
        double x, y, theta;

        if (fabs(yaw_rate) < 0.001) { // If the yaw rate is 0
            x = particle.x + velocity * delta_t * cos(particle.theta);
            y = particle.y + velocity * delta_t * sin(particle.theta);
            theta = particle.theta;

            if (debug_predict) {
                cout<<"YZ prediction ("<<x<<", "<<y<<", "<<theta<<") "<<endl;
            }

        } else { // If the yaw rate is not zero
            x = particle.x +
                (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));

            y = particle.y +
                (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));

            theta = particle.theta + yaw_rate * delta_t;

            if (debug_predict) {
                cout<<"YNZ prediction ("<<x<<", "<<y<<", "<<theta<<") "<<endl;
            }
        }

        // Add noise only if not debugging the predict function
        if (!debug_predict) {
            x += dist_x(gen);
            y += dist_y(gen);
            theta += dist_theta(gen);
        }

        // Update the particle
        particle.x = x;
        particle.y = y;
        particle.theta = theta;
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    // If debugging the predict function, don't do anything here
    if (debug_predict) {
        return;
    }

    // Loop through all the particles
    weights = vector<double>(); // Reinitialize weights

    for (auto &particle : particles) {
        // Take observations in the car's co-ord system, and transform them into
        // the map co-ord system.
        vector<LandmarkObs> observations_map;

        // Loop through all the observations for conversion
        for (auto &observation : observations) {
            double x = particle.x + observation.x * cos(particle.theta) - observation.y * sin(particle.theta);
            double y = particle.y + observation.x * sin(particle.theta) + observation.y * cos(particle.theta);

            // Observations are verified
//            cout<<"Observation ("<<observation.x<<", "<<observation.y<<") -> ("<<x<<", "<<y<<")"<<endl;

            observations_map.push_back(LandmarkObs{.id=observation.id, .x=x, .y=y});
        }

        // Get landmarks within range. This can still include extraneous points
        // because we're not calculating the euclidean distance, but just the x
        // and y distance in isolation. But it's still a big improvement compared
        // to including all the landmarks.
        vector<Map::single_landmark_s> landmarks_in_range;
        for (auto &landmark : map_landmarks.landmark_list) {
            if (fabs(particle.x - landmark.x_f) > sensor_range) {
                continue;
            }

            if (fabs(particle.y - landmark.y_f) > sensor_range) {
                continue;
            }

            landmarks_in_range.push_back(landmark);
        }

        // For each distance to land-mark from the observation, try associating it with the nearest
        // landmark from the map
        double particle_weight = 1.0;

        for (auto &observation : observations_map) {
            double minimum_distance = 0.0;
            const Map::single_landmark_s *best_landmark = nullptr;

            // Calculate euclidean distance to each map_landmark
//            cout<<"DISTANCES ";
            for (auto &landmark :landmarks_in_range) {
                double distance = sqrt(pow(observation.x - landmark.x_f, 2) + pow(observation.y - landmark.y_f, 2));
//                cout<<distance<<", ";

                if ((best_landmark == nullptr) || (distance < minimum_distance)) {
                    best_landmark = &landmark;
                    minimum_distance = distance;
                }
            }
//            cout<<endl;

            // Verify associations
            cout<<"Observation ("<<observation.x<<", "<<observation.y
                <<") -> Landmark ("<<best_landmark->x_f<<", "<<best_landmark->y_f<<")"
                <<" min dist "<<minimum_distance<<endl;


//            cout<<"min dist "<<minimum_distance<<best_landmark->id_i<<endl;
            // Calculate and update the weight
            double gauss_norm = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
            double exponent = pow(observation.x - best_landmark->x_f, 2) / (2 * pow(std_landmark[0], 2)) +
                              pow(observation.y - best_landmark->y_f, 2) / (2 * pow(std_landmark[1], 2));
            double observation_weight = gauss_norm * exp(-exponent);

            particle_weight *= observation_weight;
        }

        particle.weight = particle_weight;
        weights.push_back(particle_weight);
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // If debugging the predict function, don't do anything here
    if (debug_predict) {
        return;
    }

    default_random_engine gen;
    discrete_distribution<int> distribution(weights.begin(), weights.end());
    vector<Particle> next_particles;

    for (int i = 0; i < num_particles; i++) {
        int index = distribution(gen);
        next_particles.push_back(particles[index]);
    }

    particles = next_particles;
}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
                                         const std::vector<double> &sense_x, const std::vector<double> &sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
