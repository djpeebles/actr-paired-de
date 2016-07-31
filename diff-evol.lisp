;;; =============================================================
;;; Differential Evolution algorithm for finding model parameters
;;; =============================================================
;;;
;;; Go through each unit in the population (the "target vector")
;;;   Create a "trial vector":
;;;     Select a random vector (not the target) and add a weighted random difference vector
;;;     Mate the target vector with the random vector to create the trial vector
;;; Run the model with the parameters in the trial vector.
;;; If the output of the model is "better" than with the target vector,
;;; the trial vector replaces the target in the next generation
;;; ==========================================================================
;;;
;;; This version works for any function which can return a real numbered value
;;; as long as an appropriate parameter setting function is also provided.
;;;
;;; To run it call the optim function with parameters as described below.
;;; There are some examples of general tasks and an ACT-R model at the bottom.
;;;
;;; ==========================================================================

;;; Default DE parameters

(defparameter *ngen* 10 "number of generations")
(defparameter *NP* 25 "size of population")
(defparameter *F* 0.5 "scale factor for weighted difference computation (0<f<=1.2) optimum 0.4<f<=1.0")
(defparameter *CR* 0.5 "crossover constant in recombination computation (0<=cr<=1)")

;; =======================================================================
;; Return a real-valued random number between max and min making sure that
;; random is passed a floating point to avoid integer only results.
;; =======================================================================

(defun bounded-random (min max)
  (let ((delta (- max min)))
    (+ min (if (integerp delta)
               (random (* 1.0 delta))
             (random delta)))))

;; ===================================================================
;; For each vector in the population, initialise each dimension with
;; a uniformly distributed random number bounded by the max and min
;; provided for that dimension.
;; ===================================================================

(defun create-population (np parameters ndim)
  (let ((pop (make-array np)))
    (dotimes (i np pop)
      (let ((vector (make-array ndim :initial-element nil)))
        (dotimes (dim (length parameters))
          (let ((param (nth dim parameters)))
            (setf (svref vector dim) (bounded-random (second param) (third param)))))
        (setf (svref pop i) vector)))))

;; ===================================================================
;; Some simple helper functions to keep the other code a little easier
;; to read.
;; ===================================================================

(defun create-parameter-list (vector parameters)
  "Returns a list of parameter-spec and current value lists"
  (map 'list 'list (mapcar 'first parameters) vector))

(defun set-fitness (vector fitness-index result)
  (setf (svref vector fitness-index) result))

(defun get-fitness (vector fitness-index)
  (svref vector fitness-index))

(defun pprint-vector (vector parameters output)
  (format output "~{~{~s ~f~}~^, ~} ~@[fitness: ~f~]~%"
    (create-parameter-list vector parameters) (svref vector (length parameters))))

;; =================================================================
;; Find the vector in the population provided which has the best
;; fitness given the indicated direction (:min or :max) and return a
;; list of parameter specifications for that vector printing the
;; result to the output stream specified.
;; =================================================================

(defun get-winner (pop direction parameters fitness-index output)
  (let ((best nil) (best-index nil))
    (dotimes (vec (length pop))
      (let ((fitness (get-fitness (svref pop vec) fitness-index)))
        (when (or (null best)
                  (and (eq direction :max)
                       (> fitness best))
                  (and (eq direction :min)
                       (< fitness best)))
          (setf best fitness best-index vec))))
    (format output "~%Highest fitness: ")
    (pprint-vector (svref pop best-index) parameters output)
    (create-parameter-list (svref pop best-index) parameters)))

;; ==================================================================
;; Compute and save the fitness value in a vector using the functions
;; specified to set the parameters and assign the parameters.
;; ==================================================================

(defun compute-fitness (vector model-function assign-model-parameters parameters fitness-index)
  (funcall assign-model-parameters (create-parameter-list vector parameters))
  (let ((result (funcall model-function)))
    (if (realp result)
        (set-fitness vector fitness-index result)
      (error (format nil "Function ~s returned value ~s which is not a real number."
               model-function result)))))

;; ===================================================
;; Compute the fitness for each vector in a population
;; and output a trail of dots to indicate progress to
;; the stream provided.
;; ===================================================

(defun compute-initial-fitness (pop model-function assign-model-parameters parameters fitness-index output)
  (format output "Initial generation ")
  (dotimes (target (length pop))
    (format output ".")
    (compute-fitness (svref pop target) model-function assign-model-parameters parameters fitness-index)))

;; ====================================
;; This is the function to call to run the process.
;;
;; It requires two parameters:
;;  - The first must indicate a function which takes no parameters.
;;    That function will be called to generate a result for a trial.
;;  - A list of parameter specifications.  Each parameter specification
;;    must be a list of three items:
;;     * The first is an indicator for the parameter which will be
;;       provided to the parameter assignment function.
;;     * The second must be a real value that indicates the minimum
;;       value that parameter can have.
;;     * The third must be a real value that indicates the maximum
;;       value that parameter can have.
;;
;; There are several keyword parameters which may be provided.
;;
;; :ngen - How many generations to run.
;;         The default is the value of *ngen*.
;; :np -   The size of each generation.
;;         The default is the value of *NP*.
;; :f  -   The scale factor for the difference calculation.
;;         Must be a real value such that 0 < f <= 1.2.
;;         The default is the value of *f*.
;; :cr -   The crossover constant for creating a new vector.
;;         Must be a real value such that 0 <= cr <= 1.
;;         The default is the value of *cr*.
;; :assign-model-parameters - Must be a function which accepts one
;;         parameter.  That function will be passed a list of lists
;;         prior to calling the result function.  Each sublist
;;         will have two elements:
;;          - The first value will be a parameter indicator (as
;;            given in the parameter specification).
;;          - The second value will be the current value for that
;;            parameter in the vector for which a result is desired.
;;         If ACT-R is also loaded then it will have a default value
;;         of a function that will set ACT-R general parameters for
;;         a model after it is reset.  Otherwise, a function must be
;;         specified.
;; :direction - Either the keyword :min or :max indicating how to find
;;              the best result in a generation.  The default is :max.
;; :output - An indicator for a stream to which all of the output will
;;           be written.  The default is t, which outputs to *standard-
;;           output*.  A value of nil will suppress the output.
;;
;; This function returns a list of lists with the best parameters
;; that are found.  That list will have the same format as the one
;; passed to the assign-model-parameters function.
;;
;; ====================================

(defun optim (model-function model-parameters
              &key (ngen *ngen*)(np *NP*)(f *F*)(cr *CR*)
                (assign-model-parameters #+:act-r 'assign-act-r-parameters)
                (direction :max)
                (output t))

  ;; Safety check all the provided parameters

  (cond ((not (or (functionp model-function) (fboundp model-function)))
         (format t "Model-function parameter to optim must be a function or a symbol that names a function, but given ~s~%" model-function))
        ((not (and model-parameters (every (lambda (x) (and (= (length x) 3) (realp (second x)) (realp (third x)) (>= (third x) (second x)))) model-parameters)))
         (format t "Model-parameters parameter to optim must be a list of 3 element lists with the second and third elements specifying real valued bounds, but given ~s~%" model-parameters))
        ((not (and (integerp ngen) (plusp ngen)))
         (format t "Ngen parameter to optim must be a positive integer, but given ~s~%" ngen))
        ((not (and (integerp np) (plusp np)))
         (format t "Np parameter to optim must be a positive integer, but given ~s~%" np))
        ((not (and (realp f) (plusp f) (<= f 1.2)))
         (format t "F parameter to optim must be a real number such that 0 < f <= 1.2, but given ~s~%" f))
        ((not (and (realp cr) (<= 0.0 cr 1.0)))
         (format t "Cr parameter to optim must be a real number such that 0 <= cr <= 1.0, but given ~s~%" cr))
        ((not (or (functionp assign-model-parameters) (fboundp assign-model-parameters)))
         (format t "Assign-model-parameters parameter to optim must be a function or a symbol that names a function, but given ~s~%" assign-model-parameters))
        ((not (or (eq direction :min) (eq direction :max)))
         (format t "Direction parameter to optim must be either the keyword :min or :max, but given ~s~%" direction))
        ((not (or (eq output t) (null output) (streamp output)))
         (format t "Output parameter to optim must be a stream, t, or nil, but given ~s~%" output))
        (t ;; all good

         (let* ((ndim (length model-parameters))
                (fitness-index ndim)
                (randvec1 nil)
                (randvec2 nil)
                (randvec3 nil)
                (random-dimension nil)
                (current-pop (create-population np model-parameters (1+ ndim)))
                (next-pop (make-array np))
                (winner nil))

           (compute-initial-fitness current-pop model-function assign-model-parameters model-parameters fitness-index output)
           (get-winner current-pop direction model-parameters fitness-index output)

           (dotimes (gen ngen winner)           ; for gen generations
             (format output "~%Generation ~A " (1+ gen))
             (dotimes (target np)               ; for each target vector in the population
               (let ((trialvec (make-array (1+ ndim) :initial-element nil)))
                 (format output ".")

                 ;; Mutation -------------------------------------------------
                 ;; Pick three different vectors at random and create a trial
                 ;; vector from them.  For each dimension, subtract one randvec
                 ;; from another, multiply by the scale factor and add to
                 ;; the third randvec.  Make sure each dimension doesn't go
                 ;; beyond its max and min boundaries.

                 (loop do (setf randvec1 (random np))
                     until (not (= target randvec1)))
                 (loop do (setf randvec2 (random np))
                     until (and (not (= target randvec2))
                                (not (= randvec1 randvec2))))
                 (loop do (setf randvec3 (random np))
                     until (and (not (= target randvec3))
                                (not (= randvec1 randvec3))
                                (not (= randvec2 randvec3))))

                 (dotimes (dim ndim)
                   (setf (svref trialvec dim)
                     (min (third (nth dim model-parameters))  ;; no bigger than maximum value
                          (max (second (nth dim model-parameters)) ;; at least as big as the minimum value
                               (+ (* f (- (svref (svref current-pop randvec1) dim)
                                          (svref (svref current-pop randvec2) dim)))
                                  (svref (svref current-pop randvec3) dim))))))

                 ;; Recombination ---------------------------------------------
                 ;; crossover trial vector and target vector.  If *CR* is 0.5
                 ;; there is a 50% chance that the vector will come from either
                 ;; the target or the trial vector.  One dimension is selected
                 ;; at random and allocated a value from the trial vector in
                 ;; order to be certain that at least one of the dimensions is
                 ;; different from the target vector.  Increasing *CR*
                 ;; increases the likelihood that the dimension will come from
                 ;; the donor vector

                 (setf random-dimension (random ndim))

                 (dotimes (dim ndim)
                   (if (and (> (random 1.0) cr)
                            (not (= dim random-dimension)))
                       (setf (svref trialvec dim) (svref (svref current-pop target) dim))))

                 ;; Selection -----------------------------------------------
                 ;; Run the model using the parameters of the trial vector to
                 ;; get the fitness.  Then compare the trial fitness with the
                 ;; target vector's fitness and put the winner in the next
                 ;; population with its fitness

                 (compute-fitness trialvec model-function assign-model-parameters model-parameters fitness-index)

                 (let ((target-fitness (get-fitness (svref current-pop target) fitness-index))
                       (trial-fitness (get-fitness trialvec fitness-index)))

                   (setf (svref next-pop target)
                     (if (or (and (eq direction :max) (>= trial-fitness target-fitness))
                             (and (eq direction :min) (<= trial-fitness target-fitness)))
                         trialvec
                       (svref current-pop target))))))

             ;; Update the current population and clear the next population
             ;; for the next generation

             (setf current-pop next-pop)
             (setf winner (get-winner current-pop direction model-parameters fitness-index output)))))))

;; ===================================================================
;; Here are some test functions showing how parameters and an
;; assign-model-parameters function could be written to access slots
;; of a class to store the parameters for the fitness function to use.
;; In this case it's simply creating a vector of 5 elements from 0-1,
;; some of which are the parameters being estimated and the fitness
;; calculation is the correlation between that vector and the vector
;; [.1 .4 .3 .8 .9].
;; ===================================================================

#-:act-r (defun correlation (results data)
           (flet ((sum-list (list)
                    (let ((sum 0.0))
                      (dolist (data list sum)
                        (incf sum data))))
                  (square-list (list)
                    (let ((sum 0.0))
                      (dolist (data list sum)
                        (incf sum (* data data)))))
                  (product-list (list1 list2)
                    (let ((sum 0.0))
                      (loop
                         (when (or (null list1) (null list2)) (return sum))
                         (incf sum (* (pop list1) (pop list2))))))
                  (safe-/ (num denom)
                    (/ num
                       (if (zerop denom) 1 denom))))

             (let* ((n (length results))
                    (average-results (safe-/ (sum-list results) n))
                    (average-data (safe-/ (sum-list data) n)))
               (safe-/ (- (safe-/ (product-list results data) n)
                          (* average-results average-data))
                       (* (sqrt (- (safe-/ (square-list results) n)
                                   (* average-results average-results)))
                          (sqrt (- (safe-/ (square-list data) n)
                                   (* average-data average-data))))))))

(defclass test-vector ()
  ((s1 :accessor s1 :initform .1)
   (s2 :accessor s2 :initform .2)
   (s3 :accessor s3 :initform .3)
   (s4 :accessor s4 :initform .4)
   (s5 :accessor s5 :initform .5)))

(defvar *target* (make-instance 'test-vector))

(defun test-assign-parameters (params-list)
  (dolist (p params-list)
    (setf (slot-value *target* (first p)) (second p))))

(defun compute-test-fitness ()
  (correlation (list (s1 *target*) (s2 *target*) (s3 *target*) (s4 *target*) (s5 *target*))
               (list .1 .4 .3 .8 .9)))


#| Example run using these functions to maximize correlation

> (optim 'compute-test-fitness '((s2 0 1) (s4 0 1) (s5 0 1)) :assign-model-parameters 'test-assign-parameters)
Initial generation .........................
Highest fitness: S2 0.34222478, S4 0.7453974, S5 0.95441794 fitness: 0.9914717

Generation 1 .........................
Highest fitness: S2 0.34222478, S4 0.7453974, S5 0.95441794 fitness: 0.9914717

Generation 2 .........................
Highest fitness: S2 0.34222478, S4 0.7453974, S5 0.95441794 fitness: 0.9914717

Generation 3 .........................
Highest fitness: S2 0.39427638, S4 0.8019651, S5 0.83958864 fitness: 0.99794406

Generation 4 .........................
Highest fitness: S2 0.39427638, S4 0.8019651, S5 0.83958864 fitness: 0.99794406

Generation 5 .........................
Highest fitness: S2 0.38224456, S4 0.6851593, S5 0.76147634 fitness: 0.9981367

Generation 6 .........................
Highest fitness: S2 0.38224456, S4 0.6851593, S5 0.76147634 fitness: 0.9981367

Generation 7 .........................
Highest fitness: S2 0.41680703, S4 0.7816858, S5 0.8824596 fitness: 0.99951947

Generation 8 .........................
Highest fitness: S2 0.41680703, S4 0.7816858, S5 0.8824596 fitness: 0.99951947

Generation 9 .........................
Highest fitness: S2 0.39991656, S4 0.78860056, S5 0.8802662 fitness: 0.9999465

Generation 10 .........................
Highest fitness: S2 0.39991656, S4 0.78860056, S5 0.8802662 fitness: 0.9999465
((S2 0.39991656) (S4 0.78860056) (S5 0.8802662))
|#

;; ===================================================================
;; These variables and functions provide another example of using the
;; optim function.  In this case the parameters are 5 values from 0-1
;; which are stored directly in global variables and the calculated
;; fit is simply the root mean-squared deviation between the vector of
;; those 5 parameter values and the vector [.5,.5,.5,.5,.5].  An
;; example call and output are provided below which shows changing the
;; default settings for the algorithm itself as well.
;; ===================================================================

(defvar *test-optim-1*)
(defvar *test-optim-2*)
(defvar *test-optim-3*)
(defvar *test-optim-4*)
(defvar *test-optim-5*)

(defun rmsd (results-list data-list)
  (flet ((square-list (list)
                      (let ((sum 0.0))
                        (dolist (data list sum)
                          (incf sum (* data data)))))
         (product-list (list1 list2)
                       (let ((sum 0.0))
                         (loop
                           (when (or (null list1) (null list2)) (return sum))
                           (incf sum (* (pop list1) (pop list2)))))))

    (sqrt (/ (+ (square-list results-list) (square-list data-list)
                (* -2.0 (product-list results-list data-list)))
             (length results-list)))))


(defun compute-test-fitness-2 ()
  (rmsd (list *test-optim-1* *test-optim-2* *test-optim-3* *test-optim-4* *test-optim-5*)
        (list .5 .5 .5 .5 .5)))

(defun test-assign-parameters-2 (params-list)
  (dolist (p params-list)
    (setf (symbol-value (first p)) (second p))))

#| Example run using the test functions

> (optim 'compute-test-fitness-2
         '((*test-optim-1* 0 1.0) (*test-optim-2* 0 1.0) (*test-optim-3* 0 1.0) (*test-optim-4* 0 1.0) (*test-optim-5* 0 1.0))
         :ngen 10
         :np 5
         :f 0.6
         :cr 0.75
         :assign-model-parameters 'test-assign-parameters-2
         :direction :min)
Initial generation .....
Highest fitness: *TEST-OPTIM-1* 0.31811142, *TEST-OPTIM-2* 0.45672286, *TEST-OPTIM-3* 0.18476802, *TEST-OPTIM-4* 0.9298108, *TEST-OPTIM-5* 0.3264544 fitness: 0.2642662

Generation 1 .....
Highest fitness: *TEST-OPTIM-1* 0.24844633, *TEST-OPTIM-2* 0.69864315, *TEST-OPTIM-3* 0.27942267, *TEST-OPTIM-4* 0.44620132, *TEST-OPTIM-5* 0.56701595 fitness: 0.17820111

Generation 2 .....
Highest fitness: *TEST-OPTIM-1* 0.24844633, *TEST-OPTIM-2* 0.69864315, *TEST-OPTIM-3* 0.27942267, *TEST-OPTIM-4* 0.44620132, *TEST-OPTIM-5* 0.56701595 fitness: 0.17820111

Generation 3 .....
Highest fitness: *TEST-OPTIM-1* 0.31253648, *TEST-OPTIM-2* 0.38838577, *TEST-OPTIM-3* 0.27942267, *TEST-OPTIM-4* 0.50809914, *TEST-OPTIM-5* 0.56701595 fitness: 0.141994

Generation 4 .....
Highest fitness: *TEST-OPTIM-1* 0.31253648, *TEST-OPTIM-2* 0.38838577, *TEST-OPTIM-3* 0.27942267, *TEST-OPTIM-4* 0.50809914, *TEST-OPTIM-5* 0.56701595 fitness: 0.141994

Generation 5 .....
Highest fitness: *TEST-OPTIM-1* 0.31253648, *TEST-OPTIM-2* 0.38838577, *TEST-OPTIM-3* 0.27942267, *TEST-OPTIM-4* 0.50809914, *TEST-OPTIM-5* 0.56701595 fitness: 0.141994

Generation 6 .....
Highest fitness: *TEST-OPTIM-1* 0.4443175, *TEST-OPTIM-2* 0.43614423, *TEST-OPTIM-3* 0.2680483, *TEST-OPTIM-4* 0.4444793, *TEST-OPTIM-5* 0.37617502 fitness: 0.12601185

Generation 7 .....
Highest fitness: *TEST-OPTIM-1* 0.4443175, *TEST-OPTIM-2* 0.43614423, *TEST-OPTIM-3* 0.2680483, *TEST-OPTIM-4* 0.4444793, *TEST-OPTIM-5* 0.37617502 fitness: 0.12601185

Generation 8 .....
Highest fitness: *TEST-OPTIM-1* 0.46691763, *TEST-OPTIM-2* 0.48388287, *TEST-OPTIM-3* 0.25334325, *TEST-OPTIM-4* 0.47218397, *TEST-OPTIM-5* 0.47486845 fitness: 0.11278219

Generation 9 .....
Highest fitness: *TEST-OPTIM-1* 0.32574368, *TEST-OPTIM-2* 0.3550296, *TEST-OPTIM-3* 0.5421699, *TEST-OPTIM-4* 0.4444793, *TEST-OPTIM-5* 0.5033982 fitness: 0.10606971

Generation 10 .....
Highest fitness: *TEST-OPTIM-1* 0.59355986, *TEST-OPTIM-2* 0.5302701, *TEST-OPTIM-3* 0.46298614, *TEST-OPTIM-4* 0.46263567, *TEST-OPTIM-5* 0.46084777 fitness: 0.052855402
((*TEST-OPTIM-1* 0.59355986) (*TEST-OPTIM-2* 0.5302701) (*TEST-OPTIM-3* 0.46298614) (*TEST-OPTIM-4* 0.46263567) (*TEST-OPTIM-5* 0.46084777))

|#

;; ===================================================================
;; This section contains code to implement the parameter setting
;; capability for using general parameters in ACT-R assuming that they
;; should be set when the model gets reset.  ACT-R needs to be loaded
;; prior to loading this file for these components to be defined.
;;
;; To handle that we create a module which schedules an event to set
;; the parameters indicated at the start of the model run so that the
;; settings override those specified in the model definition itself
;; and guarantee that all other modules have completed their resetting
;; processes.  It is done that way because there's no guarantee to the
;; ordering of module resetting so if this module were to set the
;; parameters directly in its reset function they could be undone by
;; some other module as it is being reset.  Of course, there's no
;; guarantee that some other module won't also try to change them in
;; this same fashion, but none of the standard modules operate that
;; way so this is safe with respect to the normal ACT-R distribution.
;;
;; This process will not work if one uses reload instead of reset at
;; the start of a model run because the clear-all in the model file
;; will result in a fresh module which won't have the stored
;; parameters in it.
;; ===================================================================

#+:act-r
(progn
  (defstruct diff-evol-module params)

  (defun assign-act-r-parameters (params-list)
    (when (current-model)
      (if (some (lambda (x) (eq :BAD-PARAMETER-NAME (car (no-output (sgp-fct (list (first x))))))) params-list)
          (error-output "Assign-act-r-parameters passed an invalid parameter name in list: ~s" params-list)
        (setf (diff-evol-module-params (get-module :diff-evol)) (flatten params-list)))))

  (defun set-diff-evol-params (instance)
    (no-output (sgp-fct (diff-evol-module-params instance))))

  (defun reset-diff-evol-module (instance)
    (when (diff-evol-module-params instance)
      (schedule-event-now 'set-diff-evol-params :destination :diff-evol :output nil :priority :max)))

  (define-module-fct :diff-evol nil nil
    :version "1.0a1"
    :documentation "module that sets model parameters at reset for use with the Differential Evolution optimizer"
    :creation (lambda (x) (declare (ignore x)) (make-diff-evol-module))
    :reset (list nil nil 'reset-diff-evol-module)))

#| Here is an example trace of using it with the fan effect model
   from the ACT-R tutorial.

;; First load the fan effect model from the tutorial

> (actr-load "ACT-R:tutorial;unit5;fan.lisp")

;; Change the output function used in the code for the experiment to
;; just return the correlation

> (defun output-person-location (data)
   (let ((rts (mapcar 'first data)))
    (correlation rts *person-location-data*)))

;; Write a function to run the task suppressing the model output and
;; any warnings which may result from poor parameter choices

> (defun run-fan-with-no-output ()
    (let ((*standard-output* (make-string-output-stream)))
      (suppress-warnings (fan-experiment))))

;; Specify the two parameters to adjust in the model with their ranges
;; and use the default DE configuration

> (optim 'run-fan-with-no-output '((:mas 1.4 3.0) (:lf 0.1 1.5)))
Initial generation .........................
Highest fitness: :MAS 1.6094178, :LF 0.85362613 fitness: 0.863915

Generation 1 .........................
Highest fitness: :MAS 1.6094178, :LF 0.85362613 fitness: 0.863915

Generation 2 .........................
Highest fitness: :MAS 1.5723848, :LF 1.5 fitness: 0.86421466

Generation 3 .........................
Highest fitness: :MAS 1.5984622, :LF 0.14665142 fitness: 0.8647397

Generation 4 .........................
Highest fitness: :MAS 1.5984622, :LF 0.14665142 fitness: 0.8647397

Generation 5 .........................
Highest fitness: :MAS 1.5549074, :LF 0.1 fitness: 0.86499417

Generation 6 .........................
Highest fitness: :MAS 1.5723848, :LF 0.1 fitness: 0.86647576

Generation 7 .........................
Highest fitness: :MAS 1.5723848, :LF 0.1 fitness: 0.86647576

Generation 8 .........................
Highest fitness: :MAS 1.5723848, :LF 0.1 fitness: 0.86647576

Generation 9 .........................
Highest fitness: :MAS 1.5723848, :LF 0.1 fitness: 0.86647576

Generation 10 .........................
Highest fitness: :MAS 1.5723848, :LF 0.1 fitness: 0.86647576
((:MAS 1.5723848) (:LF 0.1))

;; Now change the output function to return the mean deviation
;; and see what we get when minimizing that

> (defun output-person-location (data)
    (let ((rts (mapcar 'first data)))
      (mean-deviation rts *person-location-data*)))

> (optim 'run-fan-with-no-output '((:mas 1.4 3.0) (:lf 0.1 1.5)) :direction :min :output nil)
((:MAS 1.6236947) (:LF 0.6085504))

|#

;; ===========
;; end of file
;; ===========
