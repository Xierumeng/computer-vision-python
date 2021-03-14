"""
Geolocation module to map pixel coordinates to geographical coordinates
"""

import numpy as np
import threading
import queue


class Geolocation:
    """
    Locates the geographical position of a set of pixels
    """

    # Global (static) class members

    # Stop request for multithreading
    stopRequested = False
    stopLock = threading.Lock()

    # These are constants and should not be changed while working
    cameraResolution = np.array([1000, 1000])
    referencePixels = np.array([[0, 0],
                                [0, 1000],
                                [1000, 0],
                                [1000, 1000]])

    # External accesses must be read-only
    locations = []
    locationsLock = threading.Lock()


    # TODO Class members
    def __init__(self):
        """
        Initial setup of class members

        Returns
        -------
        Geolocation
        """
        # Input to convert_input()
        # TODO input interfacing with the previous module
        # self.__rawPlaneDataStruct

        # Input to gather_point_pairs()
        self.__cameraOrigin3o = np.array([0.0, 0.0, 2.0])
        self.__cameraDirection3c = np.array([0.0, 0.0, -1.0])
        self.__cameraOrientation3u = np.array([1.0, 0.0, 0.0])
        self.__cameraOrientation3v = 1 * np.cross(self.__cameraDirection3c, self.__cameraOrientation3u)

        # Input to calculate_pixel_to_geo_mapping()
        self.__pixelToGeoPairs = np.array([[[0, 0], [0, 0]],
                                           [[1, 1], [1, 1]],
                                           [[2, 2], [2, 2]],
                                           [[3, 3], [3, 3]]])

        # Input to map_location_from_pixel()
        self.__pixelToGeoMap = np.array([1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0])

        # TODO Input to construct_spread()
        self.__centreGeoPoint = np.array([[0.0, 0.0]])
        self.__cornerGeoPoints = np.array([[1.0, 0.0],
                                           [0.0, 1.0],
                                           [-1.0, 0.0],
                                           [0.0, -1.0]])

        # Output
        self.__locationGuess = np.array([0.0, 0.0], 4.0, 0.8)

        return


    def run_locator(self, pipelineIn):
        """
        Repeatedly runs the geolocation from start to end
        until requested to stop
        No unit tests
        TODO Analysis part to add, input interfacing
        Returns
        -------

        """

        while (True):

            # Check for a stop request
            self.stopLock.acquire()
            if (self.stopRequested):
                self.stopLock.release()
                break

            self.stopLock.release()

            # TODO Get input from the previous module
            # TODO This will block forever if the pipeline is empty before a stop request!
            self.__rawPlaneDataStruct = pipelineIn.get()

            # TODO Convert inputs into something usable
            # self.convert_input()

            # Get corresponding geographical coordinates
            self.__pixelToGeoPairs = self.gather_point_pairs()

            # TODO Verify no bad collinear points
            # self.__pixelToGeoPairs = self.check_collinearity()
            if (self.__pixelToGeoPairs.shape[0] < 4):
                continue

            # Create the map
            self.__pixelToGeoMap = self.calculate_pixel_to_geo_mapping()

            # TODO Get the locations
            # cornerPixels = np.array([self.__rawPlaneDataStruct.pixel[0],
            #                          self.__rawPlaneDataStruct.pixel[1],
            #                          self.__rawPlaneDataStruct.pixel[3], # TODO Numbers
            #                          self.__rawPlaneDataStruct.pixel[4]])
            # self.__centreGeoPoint = self.map_location_from_pixel(np.array([self.__rawPlaneDataStruct.pixel[2]) # TODO
            if (self.__centreGeoPoint.shape[0] != 1):
                continue

            # self.__cornerGeoPoints = self.map_location_from_pixel(cornerPixels)

            # TODO Find the best radius around the geographical location
            # confidence = self.__rawPlaneDataStruct.confidence
            # self.__locationGuess = np.array([self.__centreGeoPoint, self.construct_spread(), confidence])

            # Insert into the list
            self.locationsLock.acquire()
            self.locations.append(self.__locationGuess)
            self.locationsLock.release()

        return


    # TODO Placeholder, add functionality once we figure out how to convert raw plane data
    def convert_input(self):
        """
        Converts plane data into data usable by Geolocation

        Returns
        -------

        """

        return

    def gather_point_pairs(self):
        """
        Outputs pixel-geographical coordinate point pairs from camera position and orientation

        Returns
        -------
        # np.array(shape=(n, 2, 2))
        """

        pixelGeoPairs = np.empty(shape=(0, 2, 2))
        minimumPixelCount = 4  # Required for creating the map
        validPixelCount = self.__referencePixels.shape[0]  # Current number of valid pixels (number of rows)
        maximumZcomponent = -0.1  # This must be lesser than zero and determines if the pixel is pointing downwards

        # Find corresponding geographical coordinate for every valid pixel
        for i in range(0, self.__referencePixels.shape[0]):

            # Not enough pixels to create the map, abort
            if (validPixelCount < minimumPixelCount):
                return np.empty(shape=(0, 2, 2))

            # Convert current pixel to vector in world space
            pixel = self.__referencePixels[i]
            # Scaling in the u, v direction
            scalar1m = 2 * pixel[0] / self.__cameraResolution[0] - 1
            scalar1n = 2 * pixel[1] / self.__cameraResolution[1] - 1

            # Linear combination formula
            pixelInWorldSpace3a = self.__cameraDirection3c + scalar1m * self.__cameraOrientation3u + scalar1n * self.__cameraOrientation3v

            # Verify pixel vector is pointing downwards
            if (pixelInWorldSpace3a[2] > maximumZcomponent):
                validPixelCount -= 1
                continue

            # Find intersection of the pixel line with the xy-plane
            x = self.__cameraOrigin3o[0] - pixelInWorldSpace3a[0] * self.__cameraOrigin3o[2] / pixelInWorldSpace3a[2]
            y = self.__cameraOrigin3o[1] - pixelInWorldSpace3a[1] * self.__cameraOrigin3o[2] / pixelInWorldSpace3a[2]

            # Insert result
            pair = np.vstack((self.__referencePixels[i], [x, y]))
            pixelGeoPairs = np.concatenate((pixelGeoPairs, [pair]))

        return pixelGeoPairs

    def calculate_pixel_to_geo_mapping(self):
        """
        Outputs transform matrix for mapping pixels to geographical points

        Returns
        -------
        np.array(shape=(3,3))
        """

        # Declare 4 matrices
        # Assign relevant values, shapes and data types
        # Create a 3x3 matrix with the coordinates as vectors with 1 as the z component => np.array([[x1, x2, x3], [y1, y2, y3], [1, 1, 1]])
        sourcePixelMatrix = np.vstack((self.__pixelToGeoPairs[0:3, 0:1].reshape(3, 2).T, [1, 1, 1])).astype(np.float64)
        sourcePixelVector = np.vstack((self.__pixelToGeoPairs[3, 0:1].reshape(1, 2).T, [1])).astype(np.float64)
        mappedGeoMatrix = np.vstack((self.__pixelToGeoPairs[0:3, 1:2].reshape(3, 2).T, [1, 1, 1])).astype(np.float64)
        mappedGeoVector = np.vstack((self.__pixelToGeoPairs[3, 1:2].reshape(1, 2).T, [1])).astype(np.float64)

        # Solve system of linear equations to get value of coefficients
        solvedPixelVector = np.linalg.solve(sourcePixelMatrix, sourcePixelVector)
        solvedGeoVector = np.linalg.solve(mappedGeoMatrix, mappedGeoVector)

        # Multiply coefficients with corresponding columns in matrices sourcePixelMatrix and mappedGeoMatrix
        for i in range(0, 3):
            sourcePixelMatrix[:, i] *= solvedPixelVector[i][0]
            mappedGeoMatrix[:, i] *= solvedGeoVector[i][0]

        # Invert sourcePixelMatrix
        # Using pinv() instead of inv() for handling ill-conditioned matrices
        sourcePixelMatrixInverse = np.linalg.pinv(sourcePixelMatrix)

        # Return matrix product of mappedGeoMatrix and sourcePixelMatrixInverse
        return (mappedGeoMatrix.dot(sourcePixelMatrixInverse))
